# CUDA RL Pong

Training pong agent using policy gradient implemented in CUDA

## Setup Docker Container

~~~bash
cd CUDAPong

sudo docker run -it --name cmpe214_project --privileged --gpus all -v $PWD:/github/CUDAPong ubuntu:16.04
~~~

## Pong Gym ZMQ Python Server & Client 

You will navigate to the gym server and client folders:

~~~bash
# Terminal 1
cd ~/github/archil/CUDAPong/example/zmq/py/simple/gym_server

# Terminal 2
cd ~/github/archil/CUDAPong/example/zmq/py/simple/gym_client
~~~

We **start the OpenAI Gym server** that waits to receive random actions (or later predicted actions) from our RL client over ZMQ using command:

~~~bash
~/github/archil/CUDAPong/example/zmq/py/simple/gym_server$ python gym_server.py
~~~

<!-- ![run_gym_server_for_pong.jpg](./images/run_gym_server_for_pong.jpg) -->

<img src="./images/run_gym_server_for_pong.jpg" width="75%" height="75%">

We then **run our OpenAI RL Gym client** for it to send random actions to our OpenAI Gym server over ZMQ using command:

~~~bash
~/github/archil/CUDAPong/example/zmq/py/simple/gym_client$ python simple_pong_client.py
~~~

We see our reward in the terminal for our steps taken based on the actions we sent the server.

<!-- ![run_gym_client_rand_actions_for_pong.jpg](./images/run_gym_client_rand_actions_for_pong.jpg) -->

<img src="./images/run_gym_client_rand_actions_for_pong.jpg" width="25%" height="25%">

As our client sends those random actions to our server, our server renders those frames of the environment into jpeg images, and then streams them as jpeg bytes back to the client for the client to animate them to the user using matplotlib:

<!-- ![stream_pong_animation_server_to_client.jpg](./images/stream_pong_animation_server_to_client.jpg) -->

<img src="./images/stream_pong_animation_server_to_client.jpg" width="50%" height="50%">

## Pong Gym ZMQ Python Server & C++ Client 

You will navigate to the gym server and client folders:

~~~bash
# Terminal 1 (Python OpenAI Gym ZMQ Server)
cd ~/github/archil/CUDAPong/example/zmq/py_cpp/simple/gym_server

# Terminal 2 (C++ ZMQ Client)
cd ~/github/archil/CUDAPong/example/zmq/py_cpp/simple/gym_client
~~~

We **start the Python OpenAI Gym Server** that waits to receive requests for the server to run a random action (or later a predicted action) from our C++ client over ZMQ using command:

~~~bash
~/github/archil/CUDAPong/example/zmq/py/simple/gym_server$ python gym_server.py
~~~

For our **C++ ZMQ Client**, we use conan install, generate the CMake build files for our OS, and build the client project:

~~~bash
cd ~/github/archil/CUDAPong/example/zmq/py_cpp/simple/gym_client

mkdir build_client && cd build_client

# path to conanfile.txt file
conan install ~/github/archil/CUDAPong

cmake ..
cmake --build . -j $(nproc)
~~~

We then **run our C++ ZMQ Client** for it to send requests for the OpenAI gym server to run random actions in pong over ZMQ using command:

~~~bash
~/github/archil/CUDAPong/example/zmq/py/simple/gym_client$ ./cpp_pong_client
~~~

Results coming soon...

## TODO: 

- [x] MVP: Playing Pong: Py Client ←ZMQ→ Py Gym Server
    - Plays pong with random actions in Python
- [ ] MVP: Playing Pong: C++ Client ←ZMQ→ Py OpenAI Gym Server (Need Testing)
    - Will play pong with random actions that the C++ client requests the Python Gym server to do
- [ ] MVP: RL Policy Gradient CUDA C++ Inference: Feed Forward (Need Testing)
- [ ] MVP: Playing Pong: C++ CUDA Client ←ZMQ→ Py OpenAI Gym Server (Needs to Integrate CUDA into Client)
    - Will perform inference with CUDA C++ policy gradient feed forward network to play pong
- [ ] MVP: Playing Pong: C++ Torch Client ←ZMQ→ Py OpenAI Gym Server (Needs to Write C++ Torch Code)
    - Will perform inference with Torch C++ policy gradient feed forward network to play pong
- [ ] MVP: Benchmark: Inference Time (Pong)
    - Will plot inference execution time benchmark plots for playing pong with CUDA C++ and Torch C++ policy gradient
- [ ] Bonus: RL Policy Gradient CUDA C++ Training :Backward Propagation (Code Still In Progress)
- [ ] Bonus: Benchmark: Training Time (Pong)
    - If we have extra time, we will finish adding custom CUDA C++ for training policy gradient, so we can benchmark
    it against SoA Torch C++ Policy Gradient training


## References

Coming soon...
