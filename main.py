import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3", render_mode="human")
# Initialise the lunar lander environment and allow us to see it.

observation = env.reset()
print("initial observation:" , observation)


# observation: what the agent can "see" - lander position, velocity...  is in the following form:
# first box are minimum values, second box are maximum values
# third box (8,) is dimension count
# float 32 is the data type
#
# Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
# This space represents [Sx,Sy,Vx,Vy,theta,V_theta,bool1,bool2]
# Sx,Sy: x and y position
# Vx,Vy: x and y velocity
# theta: Angle of the lander relative to y or x axis
# V_theta: Angular velocity, i.e. rate of spin
#bool1,bool2: Are legs 1 and/or 2 in contact with the ground


# The action space has four discrete actions:
# 0: do nothing
# 1: fire left orientation engine
# 2: fire main engine
# 3: fire right orientation engine


# Divide our observation space into segments
pos_space = np.linspace(-2.5, 2.5, 10)
vel_space = np.linspace(-2.5, 2.5, 10)
theta_space = np.linspace(-6.2831855, 6.2831855, 20)
V_theta_space = np.linspace(-10,10,10)
bool1_space = [0,1]
bool2_space = [0,1]

print("divided obs space")
# print(pos_space, vel_space, theta_space, V_theta_space, bool1_space, bool2_space)
print(env.action_space.n)
q = np.zeros((len(pos_space), len(vel_space),len(theta_space), len(V_theta_space), env.action_space.n)) # init a 10x10x20x10x4




# for _ in range(150):
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
#
#     # If the episode has ended then we can reset to start a new episode
#     if terminated or truncated:
#         observation, info = env.reset()
