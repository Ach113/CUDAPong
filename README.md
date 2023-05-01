# CUDA RL Pong

Training pong agent using policy gradient implemented in CUDA

## Gym ZMQ Server & Client 

We **start the OpenAI Gym server** that waits to receive random actions (or later predicted actions) from our RL client over ZMQ:

![run_gym_server_for_pong.jpg](./images/run_gym_server_for_pong.jpg)

We then **run our OpenAI RL Gym client** for it to send random actions to our OpenAI Gym server over ZMQ:

![run_gym_client_rand_actions_for_pong.jpg](./images/run_gym_client_rand_actions_for_pong.jpg)

As our client sends those random actions to our server, our server renders those frames of the environment into jpeg images, and then streams them as jpeg bytes back to the client for the client to animate them to the user using matplotlib:

![stream_pong_animation_server_to_client.jpg](./images/stream_pong_animation_server_to_client.jpg)

## References

Coming soon...
