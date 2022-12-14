#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:37:35 2022

@author: joachimvanneste
"""

import gym
import matplotlib.pyplot as plt 
import time 
import numpy as np

def display():
    env_screen = env.render()
    env.close()
    plt.imshow(env_screen)
    
env = gym.make('MountainCar-v0', render_mode='human')

print("Upper Bound for Env Observation", env.observation_space.high)
print("Lower Bound for Env Observation", env.observation_space.low)
print("The action space: {}".format(env.action_space.n))

def discretesize(s):
    s_adjusted = (s - env.observation_space.low)*np.array([10, 100])
    return np.round(s_adjusted, 0).astype(int)

    

def randomAgent():
    return env.action_space.sample()

# persistents - q table, frequencies  
global Q
global N
global s,a,r
global terminal 

num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
num_states = np.round(num_states, 0).astype(int) + 1

# maybe initialise to q table to zeros 
Q = np.random.uniform(low=-1, high=1, size = (num_states[0], num_states[1], env.action_space.n))
N = np.zeros(((num_states[0], num_states[1], env.action_space.n)))
# previous state, action and reward
s,a,r = None, None, None
terminal = False

#def QLearningAgent(current_state, current_reward):
obs = env.reset()
a = env.action_space.sample()
obs, reward, done, truncated, info = env.step(a)

print(obs)

print(discretesize(obs))

print()

print()

episodes = 0
obs = env.reset()
done = False
for step in range(episodes):
    while not done:
        action = randomAgent()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.001)
        
        # If the epsiode is up, then start another one
        if done:
            print("Episode {} done".format(episodes))
            env.reset()
# Close the env
env.close()