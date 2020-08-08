import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

def diff_equation(dat, t):
	theta = dat[0]
	phi = dat[1]
	d_theta = phi
	# d_phi = -1*(b/m)*phi - (g/l)*math.sin(theta)
	d_phi = -1*0.1*phi - 20*math.sin(theta)
	return [d_theta, d_phi]

initial = [2, 0]
time = np.linspace(0, 100, 2000)
dat_train = odeint(diff_equation, initial, time)
theta_train = dat_train[:,0]

plt.xlabel("x")
plt.ylabel("y")
plt.title("Damped harmonic oscillator")
plt.plot(time,theta_train)
plt.show()