install with pip install -e ligo-environment
After, you can create an instance of the environment with gym.make('ligo_environment:two-stage-v1')

An example implementation of the environment is in test.py, in the function run_two_stage
Each run of the environment has a randomly generated seismic noise using noise.py, but alternatively you can pass a predefined seismic noise into the constructor
The action space is the controlling torque on the actuated pendulum (the first pendulum). Actions of 0, 1, 2 correspond to torques of -1, 0, +1 respectively.
The friction parameter 

I attempted to implement reinforcement training in the controller module.

