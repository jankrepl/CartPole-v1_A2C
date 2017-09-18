"""Solving the CartPole-v1 environment with the Advantage Actor Critic (A2C) method """

import gym

from foo import *

__author__ = "Jan Krepl"
__email__ = "jankrepl@yahoo.com"
__license__ = "MIT"

# PARAMETERS
# Environment
number_of_episodes = 4000
render_bool = False
penalize_bad_states = (True, -100)  # By default the reward is always one as long as the game is still going
# This boolean enables to switch on a mode where states that lead immediately to the end are given NEGATIVE reward


# Critic
param_dict_critic = {'gamma': 1,
                     'HL_nodes_critic': 200,
                     'adam_learning_rate_critic': 0.005}

# Actor
param_dict_actor = {'HL_nodes_actor': 200,
                    'adam_learning_rate_actor': 0.001}

# Algorithm termination
number_of_consecutive_episodes = 5
threshold_average = 490

# INITIALIZATION
my_solver = SolverA2C(param_dict_critic, param_dict_actor)
env = gym.make('CartPole-v1')
solved = False
results = []

# MAIN ALGORITHM
for e in range(number_of_episodes):
    s_old = s_reshaper(env.reset())
    done = False
    t = 0
    while not done:
        t += 1

        if render_bool:
            env.render()

        a = my_solver.choose_action(s_old)

        s_new, r, done, _ = env.step(a)

        s_new = s_reshaper(s_new)

        # Penalize
        if penalize_bad_states[0] and done and t < 500:
            r = penalize_bad_states[1]
        # Train
        if not solved:
            my_solver.train(s_old, a, r, s_new, done and t < 500)

        s_old = s_new

    # Append results and check if solved
    results.append(t)

    if np.mean(results[-min(number_of_consecutive_episodes, e):]) > threshold_average:
        solved = True
        print('Stable solution found - no more training!!!!!!!!!!!!')
    else:
        solved = False

    print('The episode %s lasted for %s steps' % (e, t))
    my_solver.feed_most_recent_score(t)
