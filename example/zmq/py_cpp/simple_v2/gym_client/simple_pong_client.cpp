#include <iostream>
#include <zmq.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

int main() {
    // Create ZMQ context and socket
    cout << "Create ZMQ context and socket\n";
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REQ);

    // Connect to the OpenAI gym server
    cout << "Connect to the OpenAI gym server\n";
    socket.connect("tcp://localhost:5555");

    if(socket.connected()) {
        cout << "Connected to OpenAI gym server.\n";
    }
    else {
        cout << "Failed to connect to the OpenAI gym server\n";
        return 1;
    }

    // Start the game
    cout << "Start the game\n";
    string start_msg = "start";
    zmq::message_t start_request(start_msg.size());
    memcpy(start_request.data(), start_msg.c_str(), start_msg.size());
    socket.send(start_request);

    // Receive the initial game state
    cout << "Receive the initial game state\n";
    zmq::message_t state_reply;
    socket.recv(&state_reply);

    cout << "Render the game state\n";
    cout << "state_reply.size() = " << state_reply.size() << "\n";
    std::vector<uchar> state_data(state_reply.size());
    memcpy(state_data.data(), state_reply.data(), state_reply.size());

    for(const char& value: state_data) {
        cout << static_cast<int>(value) << " ";
    }
    cout << "\n";

    cv::Mat state_mat = cv::imdecode(state_data, cv::IMREAD_COLOR);
    cv::imshow("Pong", state_mat);
    cv::waitKey(1);

    // Play the game until it's over
    cout << "Play the game until it's over\n";
    bool done = false;
    while(!done) {
        // Render the game state

        // Choose the action
        cout << "Choose the action\n";
        int action = rand() % 6;

        // Send the action to the OpenAI gym server
        cout << "Send the action to the OpenAI gym server\n";
        string action_msg = to_string(action);
        zmq::message_t action_request(action_msg.size());
        memcpy(action_request.data(), action_msg.c_str(), action_msg.size());
        socket.send(action_request);

        // Receive the next game state and reward
        cout << "Receive the next game state and reward\n";
        zmq::message_t next_state_reply, reward_reply, done_reply, info_reply;

        // Decode next observation and display
        socket.recv(&next_state_reply);
        std::vector<uchar> next_state_data(next_state_reply.size());
        memcpy(next_state_data.data(), next_state_reply.data(), next_state_reply.size());
        cv::Mat next_state_mat = cv::imdecode(next_state_data, cv::IMREAD_COLOR);
        cv::imshow("Pong", next_state_mat);
        cv::waitKey(1);

        // zmq::message_t reward_reply;
        socket.recv(&reward_reply);
        int reward = stoi(string(static_cast<char*>(reward_reply.data()), reward_reply.size()));

        // Check if the game is over
        cout << "Check if the game is over\n";
        // zmq::message_t done_reply;
        socket.recv(&done_reply);
        done = *(reinterpret_cast<bool*>(done_reply.data()));
    }

    cout << "Exit the game\n";
    string exit_msg = "exit";
    zmq::message_t exit_request(exit_msg.size());
    memcpy(exit_request.data(), exit_msg.c_str(), exit_msg.size());
    socket.send(exit_request);

    return 0;
}