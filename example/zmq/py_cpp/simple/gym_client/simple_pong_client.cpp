#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <zmq.hpp>
// #include <pyzmq.h>
#include <pybind11/pybind11.h> // Include the PyBind11 headers
// #include <pybind11/pytypes.h>

namespace py = pybind11; // Add the namespace declaration for PyBind11

// using namespace cv;
using namespace std;

int main() {
    // set up zmq context and socket
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REQ);
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
        string state_str = string(static_cast<char*>(state_reply.data()), state_reply.size());
        std::cout << "Received initial state\n";
        cv::Mat state_mat = cv::Mat::zeros(210, 160, CV_8UC3);
        // Create a vector to hold the data
        std::vector<uchar> state_data(state_str.begin(), state_str.end());
        // Ensure the size of state_mat.data matches the size of state_data
        state_mat.data = state_data.data();
        state_mat.dataend = state_data.data() + state_data.size();

        // Run for 1000 episodes
        // Goal: Measure **PyTorch, *C++_Torch and **custom CUDA inference
        // Archil: PyTorch Policy Gradient Inference for Pong
        // James: Integrate custom CUDA Policy Gradient for Pong
        // Archil/James: Write code to measure exec inference time
        // Extra: C++ Torch Policy Gradient Inference for Pong
        // Hope our custom CUDA is faster
        // Both PyTorch and our code utilize 
        while(true) {
            // Render the game observation
            cv::imshow("Pong", state_mat);
            cv::waitKey(1);

            std::cout << "Displaying Pong\n";

            // TODO: Get the state (custom CUDA)
            // CPU Measure execution time
            // Perform inference
            // CPU Measure execution time

            // Send a message to Py OpenAI gym server to get the next action
            string get_action_msg = "get_action";
            zmq::message_t msg(get_action_msg.size());
            memcpy(msg.data(), get_action_msg.c_str(), get_action_msg.size());
            socket.send(msg);

            // Receive the action from Py OpenAI gym server
            zmq::message_t action_reply;
            socket.recv(&action_reply);
            string action_str = string(static_cast<char*>(action_reply.data()), action_reply.size());

            int action = 0; // Default value
            try {
                action = stoi(action_str);
                cout << "Action: " << action_str << endl;
            } catch(const std::invalid_argument& e) {
                cout << "Invalid action received: " << action_str << endl;
            }

            // Send a message to Py OpenAI gym server to perform the action and get the next game state
            string perform_action_msg = "perform_action " + action_str;
            zmq::message_t msg2(perform_action_msg.size());
            memcpy(msg2.data(), perform_action_msg.c_str(), perform_action_msg.size());
            socket.send(msg2);

            // Receive next game state, reward, done flag, info from Py OpenAI gym server
                // Deserialize the received data using PyZMQ's pyobj deserialization
            // receive next observation, reward, done flag, and info from server
            zmq::message_t next_state_msg, reward_msg, done_msg, info_msg;
            socket.recv(&next_state_msg);
            socket.recv(&reward_msg);
            socket.recv(&done_msg);
            socket.recv(&info_msg);

            // decode next observation and display
            string next_state_str = string(static_cast<char*>(next_state_msg.data()), next_state_msg.size());
            std::cout << "Received next state\n";
            cv::Mat next_state_mat = cv::Mat::zeros(210, 160, CV_8UC3);
            // Create a vector to hold the data
            std::vector<uchar> next_state_data(next_state_str.begin(), next_state_str.end());
            // Ensure the size of state_mat.data matches the size of state_data
            next_state_mat.data = next_state_data.data();
            next_state_mat.dataend = next_state_data.data() + next_state_data.size();
            cv::imshow("Pong", next_state_mat);
            cv::waitKey(1);

            // check if episode is done
            bool done = *(reinterpret_cast<bool*>(done_msg.data()));

            // Check if the game is over
            if(done) {
                std::cout << "Reached Done for Pong\n";
                // Send a restart message to Py OpenAI gym server to restart the game
                string restart_msg = "new_game";
                zmq::message_t msg(restart_msg.size());
                memcpy(msg.data(), restart_msg.c_str(), restart_msg.size());
                socket.send(msg);

                // Receive Pong initial observation from Py OpenAI gym server
                zmq::message_t state_reply;
                socket.recv(&state_reply);
                string state_str = string(static_cast<char*>(state_reply.data()), state_reply.size());

                std::cout << "Received initial state again\n";
                cv::Mat state_mat = cv::Mat::zeros(210, 160, CV_8UC3);
                // Create a vector to hold the data
                std::vector<uchar> state_data(state_str.begin(), state_str.end());
                // Ensure the size of state_mat.data matches the size of state_data
                state_mat.data = state_data.data();
                state_mat.dataend = state_data.data() + state_data.size();
                cv::imshow("Pong", state_mat);
                cv::waitKey(1);
            }

            // Update game state
            state_mat = next_state_mat;
        }

    } catch (const zmq::error_t& ex) {
        cerr << "Zmq error: " << ex.what() << endl;
    }

    // Close the window and clean up
    cv::destroyAllWindows();
    socket.close();
    context.close();

    return 0;
}
