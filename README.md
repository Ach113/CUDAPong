# CUDA RL Pong

Training pong agent using policy gradient implemented in CUDA

## Gym ZMQ Server & Client 

You will navigate to the gym server and client folders:

~~~bash
# Terminal 1
cd ~/github/archil/CUDAPong/example/gym_server

# Terminal 2
cd ~/github/archil/CUDAPong/example/gym_client
~~~

We **start the OpenAI Gym server** that waits to receive random actions (or later predicted actions) from our RL client over ZMQ using command:

~~~bash
~/github/archil/CUDAPong/example/gym_server/zmq$ python gym_server.py
~~~

<!-- ![run_gym_server_for_pong.jpg](./images/run_gym_server_for_pong.jpg) -->

<img src="./images/run_gym_server_for_pong.jpg" width="75%" height="75%">

We then **run our OpenAI RL Gym client** for it to send random actions to our OpenAI Gym server over ZMQ using command:

~~~bash
~/github/archil/CUDAPong/example/gym_client/py/zmq$ python rl_client.py
~~~

We see our reward in the terminal for our steps taken based on the actions we sent the server.

<!-- ![run_gym_client_rand_actions_for_pong.jpg](./images/run_gym_client_rand_actions_for_pong.jpg) -->

<img src="./images/run_gym_client_rand_actions_for_pong.jpg" width="25%" height="25%">

As our client sends those random actions to our server, our server renders those frames of the environment into jpeg images, and then streams them as jpeg bytes back to the client for the client to animate them to the user using matplotlib:

<!-- ![stream_pong_animation_server_to_client.jpg](./images/stream_pong_animation_server_to_client.jpg) -->

<img src="./images/stream_pong_animation_server_to_client.jpg" width="50%" height="50%">

## References

Coming soon...
