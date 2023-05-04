import gym
import numpy as np
import zmq

import io
from PIL import Image
import matplotlib.pyplot as plt


context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# render_mode="human"
env = gym.make('Pong-v0')
# env.metadata['render.modes'] = ['rgb_array'] # human
# env.metadata['render.fps'] = 60
state = env.reset()

done = False
while not done:
    # Choose a random action
    action = np.random.randint(env.action_space.n)
    
    # Send the action to the server
    socket.send_pyobj(action)
    
    # Receive the new state and reward from the server
    state, reward, done, info, jpeg_bytes_value = socket.recv_pyobj()
    state = np.array(state)

    # Received JPEG image over ZMQ and decode
    image = np.array(Image.open(io.BytesIO(jpeg_bytes_value)))
    
    # Render the game
    # env.render()

    # Display the image using matplotlib
    plt.imshow(image)
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()    
    print("Reward = {}".format(reward))

env.close()
socket.close()

