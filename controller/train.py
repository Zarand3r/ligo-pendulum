import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import noise

ENV_NAME = 'ligo_environment:two-stage-v1'
seismic = noise.generate_noise()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME, seismic=seismic)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
# model.add(Flatten(input_shape=(6, 1) + env.observation_space.shape))
model.add(Flatten(input_shape=(1,6,1)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# # Add a LSTM layer with 128 internal units.
# model.add(keras.layers.LSTM(128))
# # Add a Dense layer with 10 units.
# model.add(keras.layers.Dense(10))

print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
dqn.fit(env, nb_steps=50000, visualize=True, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights('saved_model/dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)

