#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <zmq.hpp>

// using namespace cv;
using namespace std;

int main() {
    // set up zmq context and socket
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REG);
    socket.connect("tcp://localhost:5555");

    try {

        // Send a message to Py OpenAI gym server to start the game
        string start_msg = "start";
        zmq::message_t msg(start_msg.size());
        memcpy(msg.data(), start_msg.c_str(), start_msg.size());
        socket.send(msg);

        // Receive Pong initial observation from Py OpenAI gym server
        zmq::message_t state_reply;
        socket.recv(&state_reply);
        string state_str = string(static_cast<char*>(state_reply.data(), state_reply.size()));
        cv::Mat state = cv::Mat::zeros(210, 160, CV_8UC3);
        memcpy(state.data, state_str.c_str(), state_str.size());

        while(true) {
            // Render the game observation
            cv::imshow("Pong", state);
            cv::waitKey(1);

            // Send a message to Py OpenAI gym server to get the next action
            string get_action_msg = "get_action";
            zmq::message_t msg(get_action_msg.size());
            memcpy(msg.data(), get_action_msg.c_str(), get_action.msg.size());
            socket.send(msg);

            // Receive the action from Py OpenAI gym server
            zmq::message_t action_reply;
            socket.recv(&action_reply);
            string action_str = string(static_cast<char*>(action_replay.data()), action_reply.size());
            int action = stoi(action_str);

            cout << "Action: " << action << endl;

            // Send a message to Py OpenAI gym server to perform the action and get the next game state
            string perform_action_msg = "perform_action " + action_str;
            zmq::message_t msg2(perform_action_msg.size());
            memcpy(msg2.data(), perform_action_msg.c_str(), perform_action_msg.size());
            socket.send(msg2);

            // Receive next game state, reward, done flag, info from Py OpenAI gym server
                // Deserialize the received data using PyZMQ's pyobj deserialization
            zmq::message_t step_reply;
            socket.recv(step_reply);
            py::tuple data = py::reinterpret_borrow<py::tuple>(PyZMQ_get(new PyZMQ_message_t(step_reply), "msgpack", "binary"));
            // Extract the next_state, reward, done flag, info dictionary from Py OpenAI gym server
            auto next_state = py::reinterpret_borrow<py::array>(data[0]);
            float reward = py::float_(data[1]);
            bool done = py::bool_(data[2]);
            py::dict info py::reinterpret_borrow<py::dict>(data[3]);

            cv::Mat next_state_mat(next_state.shape(0), next_state.shape(1), CV_8UC3, next_state.mutable_data());

            cv::imshow("Pong", next_state_mat);
            cv::waitKey(1);

            // Check if the game is over
            if(done) {
                // Send a restart message to Py OpenAI gym server to restart the game
                string restart_msg = "new_game";
                zmq::message_t msg(restart_msg.size());
                memcpy(msg.data(), restart_msg.c_str(), restart_msg.size());
                socket.send(msg);

                // Receive Pong initial observation from Py OpenAI gym server
                zmq::message_t state_reply;
                socket.recv(&state_reply);
                string state_str = string(static_cast<char*>(state_reply.data(), state_reply.size()));
                cv::Mat state_mat = cv::Mat::zeros(210, 160, CV_8UC3);
                memcpy(state_mat.data, state_str.c_str(), state_str.size());
                cv::imshow("Pong", state_mat);
                cv::waitKey(1);
            }

            // Update game state
            state = next_state;
        }

    } catch (const zmq::error_t& ex) {
        cerr << "Zmq error: " << ex.what() << endl;
    }

    // Close the window and clean up
    destroyAllWindows();
    socket.close();
    context.close();

    return 0;
}