"""
Original file: cartpole.py
Modifications by Fredrik Gustafsson
"""

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
import numpy as np
from scipy.integrate import ode
sin = np.sin
cos = np.cos
import random

import sys
import pathlib
directory = pathlib.Path(__file__).parent.absolute()
sys.path.insert(1, str(directory)+'/utils')
import noise

class TwoStage1(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	def __init__(self, seismic=None):
		self.g = -9.81 # gravity constant
		self.m0 = 1.0 # mass of cart
		self.m1 = 0.5 # mass of pole 1
		self.m2 = 0.5# mass of pole 2
		self.L1 = 1 # length of pole 1
		self.L2 = 1 # length of pole 2
		self.l1 = self.L1/2 # distance from pivot point to center of mass
		self.l2 = self.L2/2 # distance from pivot point to center of mass
		self.I1 = self.m1*(self.L1^2)/12 # moment of inertia of pole 1 w.r.t its center of mass
		self.I2 = self.m2*(self.L2^2)/12 # moment of inertia of pole 2 w.r.t its center of mass
		self.tau = 0.02  # seconds between state updates
		self.counter = 0
		self.seismic = seismic
		if seismic is None:
			self.seismic = noise.generate_noise(fs=4000, amplify=100)
		# self.seismic = noise.generate_noise()

		# Angle at which to fail the episode
		#self.theta_threshold_radians = 12 * 2 * math.pi / 360
		# # (never fail the episode based on the angle)
		self.theta_threshold_radians = 100000 * 2 * math.pi / 360
		self.x_threshold = 2.4
		self.x2_threshold = 1

		# Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
		high = np.array([
			self.x_threshold * 2,
			np.finfo(np.float32).max,
			self.theta_threshold_radians * 2,
			np.finfo(np.float32).max])

		self.action_space = spaces.Discrete(3)
		self.observation_space = spaces.Box(-high, high, dtype=np.float32)

		self.viewer = None

		# Just need to initialize the relevant attributes
		self.configure()

	def configure(self, display=None):
		self.display = display 

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		u = 0
		# u = noise.sample_seismic(self.seismic, self.counter)
		u = noise.sample_seismic_derivative(self.seismic, self.counter)
		self.counter += 1
		
		# (state_dot = func(state))
		def func(t, state, u):
			x = state.item(0)
			theta = state.item(1)
			phi = state.item(2)
			x_dot = state.item(3)
			theta_dot = state.item(4)
			phi_dot = state.item(5)
			state = np.matrix([[x],[theta],[phi],[x_dot],[theta_dot],[phi_dot]]) # this is needed for some weird reason
			
			d1 = self.m0 + self.m1 + self.m2
			d2 = self.m1*self.l1 + self.m2*self.L1
			d3 = self.m2*self.l2
			d4 = self.m1*pow(self.l1,2) + self.m2*pow(self.L1,2) + self.I1
			d5 = self.m2*self.L1*self.l2
			d6 = self.m2*pow(self.l2,2) + self.I2
			f1 = (self.m1*self.l1 + self.m2*self.L1)*self.g 
			f2 = self.m2*self.l2*self.g    
			
			D = np.matrix([[d1, d2*cos(theta), d3*cos(phi)], 
					[d2*cos(theta), d4, d5*cos(theta-phi)],
					[d3*cos(phi), d5*cos(theta-phi), d6]])
			
			C = np.matrix([[0, -d2*sin(theta)*theta_dot, -d3*sin(phi)*phi_dot],
					[0, 0, d5*sin(theta-phi)*phi_dot],
					[0, -d5*sin(theta-phi)*theta_dot, 0]])
					
			G = np.matrix([[0], [-f1*sin(theta)], [-f2*sin(phi)]])
			
			H  = np.matrix([[1],[0],[0]])
			
			I = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
			O_3_3 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
			O_3_1 = np.matrix([[0], [0], [0]])
			
			A_tilde = np.bmat([[O_3_3, I],[O_3_3, -np.linalg.inv(D)*C]])
			B_tilde = np.bmat([[O_3_1],[np.linalg.inv(D)*H]])
			W = np.bmat([[O_3_1],[np.linalg.inv(D)*G]])
			state_dot = A_tilde*state + B_tilde*u + W  
			return state_dot
		
		solver = ode(func) 
		solver.set_integrator("dop853") # (Runge-Kutta)
		solver.set_f_params(u)
		t0 = 0
		solver.set_initial_value(self.state, t0)
		solver.integrate(self.tau)
		state=solver.y
		action = action -1 # or use AVAIL_TORQUE = [-1., 0., +1]
		state[4] += self.tau*action #update theta_dot with the torque
		# state[0] = noise.sample_seismic(self.seismic, self.counter)
		# # state[3] = noise.sample_seismic(self.seismic, self.counter)
		x = state.item(0)
		theta = state.item(1)
		phi = state.item(2)
		self.state = state

		x1 = self.L1 * math.sin(abs(theta-math.pi)) 	
		x2 = x1 + (self.L2 * math.sin(abs(theta-math.pi)))	
		
		done =  abs(x) > self.x_threshold \
				or abs(x2) > self.x2_threshold \
				or self.counter > 1000 \
				# or theta > 90*2*np.pi/360 \
				# or theta < -90*2*np.pi/360 
		done = bool(done)

		cost = 10*normalize_angle(theta) + \
				10*normalize_angle(phi)
				
		reward = -cost
		
		return self.state, reward, done, {}

	def reset(self):
		# self.state = np.matrix([[0],[np.random.uniform(-0.1,0.1)],[0],[0],[0],[0]])
		self.state = np.matrix([[0],[math.pi],[math.pi],[0],[0],[0]])
		self.counter = 0
		return self.state

	def render(self, mode='human'):
		screen_width = 800
		screen_height = 600

		world_width = self.x_threshold*2
		scale = screen_width/world_width
		#carty = 100 # TOP OF CART
		# #
		carty = 300 # TOP OF CART
		polewidth = 10.0
		polelen = scale * 0.8
		cartwidth = 50.0
		cartheight = 30.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			axleoffset =cartheight/4.0
			cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
			pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			pole.set_color(.8,.6,.4)
			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)
			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth/2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5,.5,.8)
			self.viewer.add_geom(self.axle)
			
			l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
			pole2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			pole2.set_color(.2,.6,.4)
			self.poletrans2 = rendering.Transform(translation=(0, polelen-5))
			pole2.add_attr(self.poletrans2)
			pole2.add_attr(self.poletrans)
			pole2.add_attr(self.carttrans)
			self.viewer.add_geom(pole2)
			self.axle2 = rendering.make_circle(polewidth/2)
			self.axle2.add_attr(self.poletrans2)
			self.axle2.add_attr(self.poletrans)
			self.axle2.add_attr(self.carttrans)
			self.axle2.set_color(.1,.5,.8)
			self.viewer.add_geom(self.axle2)
			
			self.track = rendering.Line((0,carty), (screen_width,carty))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

		state = self.state
		cartx = state.item(0)*scale+screen_width/2.0 # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(-state.item(1))
		self.poletrans2.set_rotation(-(state.item(2)-state.item(1)))

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
		
def normalize_angle(angle):
	"""
	3*pi gives -pi, 4*pi gives 0 etc, etc. (returns the negative difference
	from the closest multiple of 2*pi)
	"""
	normalized_angle = abs(angle)
	normalized_angle = normalized_angle % (2*np.pi)
	if normalized_angle > np.pi:
		normalized_angle = normalized_angle - 2*np.pi
	normalized_angle = abs(normalized_angle)
	return normalized_angle


# # Add noise to the force action
# if self.torque_noise_max > 0:
#     torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)