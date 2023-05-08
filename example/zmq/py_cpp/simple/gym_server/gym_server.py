import gym
import zmq
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

env = gym.make('PongNoFrameskip-v4', render_mode='rgb_array')

while True:
    # Wait for a message from the client to start the game
    msg = socket.recv()
    print("start? msg = {}".format(msg))
    if msg == b'start':
        # Reset the environment and get the initial game state
        obs = env.reset()
        # Send the initial game state to the client
        socket.send_pyobj(obs)
        done = False
        while not done:
            # Wait for a message from the client to get the next action
            msg = socket.recv()
            print("get_action? msg = {}".format(msg))
            if msg == b'get_action':
                # Choose an action using the environmet's action space
                action = np.random.randint(env.action_space.n)
                print("random? action = {}".format(action))
                # Send the action to the client
                socket.send_pyobj(action)
                # Wait for a message fro the client to perform the action
                msg = socket.recv()
                print("perform_action? msg = {}".format(msg))
                if msg == b'perform_action':
                    # Perform the action and get the next game state and reward
                    obs, reward, done, info = env.step(action)
                    # Send the next obs to the client
                    socket.send_pyobj(obs)
                    # Send the reward to the client
                    socket.send_pyobj(reward)
                    # Send the done to the client
                    socket.send_pyobj(done)
                    socket.send_pyobj(info)
                    # If the game is over, wait for a message from the client to start a new game
                    if done:
                        msg = socket.recv()
                        print("new_game? msg = {}".format(msg))
                        if msg == b'new_game':
                            obs = env.reset()
                            socket.send_pyobj(obs)
    else:
        # Invalid message received, ignore and wait for the next message
        pass        