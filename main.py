import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
# Initialise the lunar lander environment and allow us to see it.

observation = env.reset()
# observation: what the agent can "see" - lander position, velocity...  is in the following form:
# Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
# This space represents [Sx,Sy,Vx,Vy,theta,V_theta,bool1,bool2]
# Sx,Sy: x and y position
# Vx,Vy: x and y velocity
# theta: Angle of the lander relative to y or x axis
# V_theta: Angular velocity, i.e. rate of spin
#bool1,bool2: Are legs 1 and/or 2 in contact with the ground

print("initial observation:" , observation)

# The action space has four discrete actions:
# 0: do nothing
# 1: fire left orientation engine
# 2: fire main engine
# 3: fire right orientation engine



for _ in range(150):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()