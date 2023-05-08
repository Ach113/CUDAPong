#include <iostream>
#include <zmq.hpp>
#include <opencv2/opencv.hpp>
#include <numpy/arrayobject.h>
#include <Python.h>

using namespace cv;
using namespace std;

int main() {
    // Initialize Python interpreter
    cout << "Initialize Python interpreter\n";
    Py_Initialize();
    import_array();

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
    cout << "Ran socket.recv(state_reply)\n";
    cout << "state_reply.size() = " << state_reply.size() << "\n";
    PyArrayObject* state_array = reinterpret_cast<PyArrayObject*>(state_reply.data());
    cout << "Converting state_reply.data to PyArrayObject*\n";
    auto state_mat = PyArray_GETCONTIGUOUS(state_array);
    cout << "Stored PyArray into state_mat\n";

    // Play the game until it's over
    cout << "Play the game until it's over\n";
    bool done = false;
    while(!done) {
        // Render the game state
        cout << "Render the game state\n";
        cv::Mat image(PyArray_SHAPE(state_array)[0], PyArray_SHAPE(state_array)[1], CV_8UC3, PyArray_DATA(state_array));
        cv::imshow("Pong", image);
        cv::waitKey(10);

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
        zmq::message_t next_state_reply;
        socket.recv(&next_state_reply);
        PyArrayObject* next_state_array = reinterpret_cast<PyArrayObject*>(next_state_reply.data());
        auto next_state_mat = PyArray_GETCONTIGUOUS(next_state_array);

        zmq::message_t reward_reply;
        socket.recv(&reward_reply);
        int reward = stoi(string(static_cast<char*>(reward_reply.data()), reward_reply.size()));

        // Check if the game is over
        cout << "Check if the game is over\n";
        zmq::message_t done_reply;
        socket.recv(&done_reply);
        done = stoi(string(static_cast<char*>(done_reply.data()), done_reply.size()));

        // Update the game state
        cout << "Update the game state\n";
        Py_XDECREF(state_array);
        state_array = next_state_array;
        state_mat = next_state_mat;
    }

    cout << "Exit the game\n";
    string exit_msg = "exit";
    zmq::message_t exit_request(exit_msg.size());
    memcpy(exit_request.data(), exit_msg.c_str(), exit_msg.size());
    socket.send(exit_request);

    // Clean up and exit
    cout << "Clean up and exit\n";
    Py_XDECREF(state_array);
    Py_XDECREF(state_mat);
    Py_Finalize();

    return 0;
}