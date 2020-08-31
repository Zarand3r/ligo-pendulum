import gym
import math

def run_one_stage(signal=None):
	env = gym.make('ligo_environment:one-stage-v0')
	env.reset()
	for t in range(100):
	    env.render()
	    if signal:
	    	env.step(signal(t))
	    else:
	    	env.step(env.action_space.sample()) # take a random action
	env.close()
	# import gym
	# env = gym.make('CartPole-v0')
	# for i_episode in range(20):
	#     observation = env.reset()
	#     for t in range(100):
	#         env.render()
	#         print(observation)
	#         action = env.action_space.sample()
	#         observation, reward, done, info = env.step(action)
	#         if done:
	#             print("Episode finished after {} timesteps".format(t+1))
	#             break
	# env.close()

def run_one_stage1(signal=None):
	env = gym.make('ligo_environment:one-stage-v1')
	env.reset()
	for t in range(100):
	    env.render()
	    if signal is None:
	    	action = env.action_space.sample()
	    else:
	    	# action = signal
	    	action = t%2
	    observation, reward, done, info = env.step(action)
	    if done:
	    	print("Episode finished after {} timesteps".format(t+1))
	    	break
	env.close()


def run_two_stage(signal=None):
	env = gym.make('ligo_environment:two-stage-v0')
	env.reset()
	for t in range(100):
	    env.render()
	    if signal is None:
	    	action = env.action_space.sample()
	    else:
	    	# action = signal
	    	action = t%2
	    observation, reward, done, info = env.step(action)
	    if done:
	    	print("Episode finished after {} timesteps".format(t+1))
	    	break
	env.close()

def run_two_stage1(signal=None):
	env = gym.make('ligo_environment:two-stage-v1')
	env.reset()
	for t in range(200):
	    env.render()
	    if signal is None:
	    	action = env.action_space.sample()
	    	# action = 1
	    else:
	    	# action = signal
	    	action = t%2
	    observation, reward, done, info = env.step(action)
	    if done:
	    	print("Episode finished after {} timesteps".format(t+1))
	    	break
	env.close()


if __name__ == '__main__':
	# run_one_stage1(0)
	run_two_stage1()

