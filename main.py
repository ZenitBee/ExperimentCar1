import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# env = gym.make("LunarLander-v3", render_mode="human")
# Initialise the lunar lander environment and allow us to see it.

# observation = env.reset()
# print("initial observation:" , observation)


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



def run(episodes, is_training=True, render=False):
    # This is our main loop

    env = gym.make("LunarLander-v3", render_mode="human")

    # Divide our observation space into segments
    x_pos_space = np.linspace(-2.5, 2.5, 8)
    y_pos_space = np.linspace(-2.5, 2.5, 8)
    x_vel_space = np.linspace(-2.5, 2.5, 8)
    y_vel_space = np.linspace(-2.5, 2.5, 8)
    theta_space = np.linspace(-6.2831855, 6.2831855, 24)
    V_theta_space = np.linspace(-10, 10, 8)
    bool1_space = [0, 1]
    bool2_space = [0, 1] #these two are not being used

    # print("divided obs space")
    q = np.zeros((len(x_pos_space),len(y_pos_space), len(x_vel_space),len(y_vel_space), len(theta_space), len(V_theta_space),
                  4))  # init a ..(insert dims) array

    episodes = episodes
    # number of discrete episodes that will be run by our agent

    learning_rate_a = 0.9
    # alpha or learning rate. bounded between 0 and 1, governs how much the values in the Q table are changed by new estimates.

    discount_factor_g = 0.9
    # gamma or discount factor. The value placed on future rewards.

    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 2 / episodes  # epsilon decay rate
    # Epsilon governs how greedy our agent is when selecting agents.

    # This is the exploration vs exploitation dichotomy
    # Typically this rate decays from being very high (i.e random) to low (i.e greedy),
    # letting our agent benefit by learning from the richer exploration of the early runs,
    # before locking in and trying to maximise rewards in the later runs.

    rng = np.random.default_rng()
    # random number generator compared with the value Epsilon to determine greed.

    rewards_per_episode = np.zeros(episodes)
    # sets up a matrix for our rewards, to be used in the bellman equation


    for i in range(episodes):

        print("current episode is", i)
        state = env.reset()  # Starting position
        state_x_pos = np.digitize(state[0][0], x_pos_space)
        state_y_pos = np.digitize(state[0][1], y_pos_space)
        state_x_vel = np.digitize(state[0][2], x_vel_space)
        state_y_vel = np.digitize(state[0][3], y_vel_space)
        state_theta = np.digitize(state[0][4], theta_space)
        state_V_theta = np.digitize(state[0][5], V_theta_space)

        # by using np.digitize() method, we are able to get the array of indices of the bin of each value,
        # which belongs to an array by using this method.

        terminated = False  # True when reached goal
        truncated = False

        rewards = 0

        while (not terminated or truncated and rewards > -1000):

            if is_training and rng.random() < epsilon:
                # Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = env.action_space.sample()
                # print("we've gone random")
            else:
                action = np.argmax(q[state_x_pos, state_y_pos, state_x_vel, state_y_vel, state_theta, state_V_theta])
                # action = np.argmax(q[state_pos, state_vel, state_theta, state_V_theta] , axis=1)
                # if action > 3:
                #     action = 0
            # print("action:", action)
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state_x_pos = np.digitize(new_state[0], x_pos_space)
            new_state_y_pos = np.digitize(new_state[1], y_pos_space)
            new_state_x_vel = np.digitize(new_state[2], x_vel_space)
            new_state_y_vel = np.digitize(new_state[3], y_vel_space)
            new_state_theta = np.digitize(new_state[4], theta_space)
            new_state_V_theta = np.digitize(new_state[5], V_theta_space)

            if is_training:
                # print("updating q values - via Bellman equation")
                # print("our reward is:", reward)
                # print("indexing into q")
                # print(state_x_pos,state_y_pos, state_x_vel, state_y_vel, state_theta, state_V_theta, action)
                q[state_x_pos,state_y_pos, state_x_vel, state_y_vel, state_theta, state_V_theta, action] = q[state_x_pos,state_y_pos, state_x_vel, state_y_vel, state_theta, state_V_theta, action] + learning_rate_a * (
                        reward + discount_factor_g * np.max(q[new_state_x_pos,new_state_y_pos, new_state_x_vel, new_state_y_vel, new_state_theta, new_state_V_theta :]) - q[
                    state_x_pos,state_y_pos, state_x_vel, state_y_vel, state_theta, state_V_theta, action]
                )

            state = new_state
            state_x_pos = new_state_x_pos
            state_y_pos = new_state_y_pos
            state_x_vel = new_state_x_vel
            state_y_vel = new_state_y_vel
            state_theta = new_state_theta
            state_V_theta = new_state_V_theta

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards
        print("rewards this episode were", rewards)


    env.close()

    # Save Q table to file
    # if is_training:
    #     f = open('lunarlander.pkl', 'wb')
    #     # pickle.dump(q, f)
    #     f.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(mean_rewards)
    plt.savefig(f'lunarlander.png')


if __name__ == '__main__':
    run(50, is_training=True, render=False)

    # run(10, is_training=False, render=True)










