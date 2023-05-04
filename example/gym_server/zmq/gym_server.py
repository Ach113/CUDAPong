import zmq
import gym
import gym.spaces
import numpy as np
import time

from PIL import Image
import io

class GymServer:
    def __init__(self, port):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:%s" % port)
        self.env = gym.make('PongNoFrameskip-v4', render_mode='rgb_array')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        # Perform a random action received from client, get new obs, reward, done
        obs, reward, done, info, _ = self.env.step(action)
        obs = np.array(obs)

        # Use 'rgb_array' mode to render the environment
        image = self.env.render()

        # Encode image as JPEG and send over ZMQ
        jpeg_bytes = io.BytesIO()
        Image.fromarray(image).save(jpeg_bytes, format='jpeg')
        # self.socket.send(jpeg_bytes.getvalues())

        # Ensure only 5 values are being unpacked
        assert len((obs, reward, done, info, jpeg_bytes.getvalue())) == 5

        # Do something with the returned values
        # print("Observation:", obs)
        # print("Reward:", reward)
        # print("Done flag:", done)
        # print("Info:", info)
        return obs, reward, done, info, jpeg_bytes.getvalue()

    def close(self):
        self.env.close()

if __name__ == '__main__':
    server = GymServer(5555)
    obs = server.reset()
    while True:
        message = server.socket.recv_pyobj()
        action = message
        obs, reward, done, info, jpeg_bytes_value = server.step(action)
        server.socket.send_pyobj((obs, reward, done, info, jpeg_bytes_value))
        if done:
            obs = server.reset()

