import gym
import zmq

# Create the OpenAI Gym environment
print("Create the OpenAI Gym environment")
env = gym.make('PongNoFrameskip-v4', render_mode='rgb_array')

# Create a ZMQ socket
print("Create a ZMQ socket")
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    # Wait for a message from the C++ client
    print("Wait for a message from the C++ client")
    message = socket.recv()

    if message == b"start":
        # Start a new episode
        print("Start a new episode")
        observation = env.reset()
        done = False

        print("observation[0] = {}".format(observation[0]))
        print("len(observation[0]) {}".format(len(observation[0])))

        # Send the observation to the C++ client
        print("Send the initial observation to the C++ client")
        socket.send_pyobj(observation[0])

        while not done:
            # Render the environment (optional)
            # print("Render the environment (optional)")
            # env.render()

            # Convert the observation to a string
            # print("Convert the observation to a string")
            # obs_str = observation.tostring()

            # Receive the action from the C++ client
            print("Receive the action from the C++ client")
            action = int(socket.recv())
            print("action = {}".format(action))

            # Perform the action in the environment
            print("Perform the action in the environment")
            observation, reward, done, _ = env.step(action)
            print("observation = {}".format(observation))
            print("reward = {}".format(reward))

            # Send the observation to the C++ client
            print("Send the next observation to the C++ client")
            print("observation[0] = {}".format(observation[0]))
            print("len(observation[0]) {}".format(len(observation[0])))
            socket.send_pyobj(observation)

            print("Send the reward to the C++ client")
            socket.send_pyobj(reward)

            print("Send the done flag to the C++ client")
            socket.send_pyobj(done)

        # Send a message indicating the end of the episode
        # print("Send a message indicating the end of the episode")
        # socket.send(b"done")

    elif message == b"exit":
        # Exit the server loop
        print("Exit the server loop")
        break

# Close the ZMQ socket and destroy the context
socket.close()
context.term()

# Closee the OpenAI Gym environment
env.close()
