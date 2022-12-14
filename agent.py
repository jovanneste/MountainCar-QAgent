#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:37:35 2022

@author: joachimvanneste
"""

import gym
env = gym.make('MountainCar-v0', render_mode='human')
import matplotlib.pyplot as plt 
import time 
import numpy as np


# persistents - q table, frequencies, prev state, action and reward
global Q
global N
global s,a,r

num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
num_states = np.round(num_states, 0).astype(int) + 1

Q = np.random.uniform(low=0, high=1, size = (num_states[0], num_states[1], env.action_space.n))
N = np.zeros(((num_states[0], num_states[1], env.action_space.n)))
# previous state, action and reward
s,a,r = None, None, 0

# learning rate
alpha = 0.3
# discount rate 
gamma = 0.9

EPISODES = 10

def display():
    env_screen = env.render()
    env.close()
    plt.imshow(env_screen)
    

def discretize(s):
    if s is None:
        return [None, None]
    if np.issubdtype(s[0], int):
        return s
    s_adjusted = (s - env.observation_space.low)*np.array([10, 100])
    return np.round(s_adjusted, 0).astype(int)


epsilon = 0.5
reduction = epsilon/EPISODES
def epsilonGreedy(s):
    global epsilon, reduction
    if np.random.random() < 1-epsilon:
         action = np.argmax(Q[s[0], s[1]])
    else:
        action = np.random.randint(0, env.action_space.n)
    
    if epsilon>0:
        epsilon-=reduction
        
    return action


def QLearningAgent(current_state, current_reward, done):
    global s,a,r,Q
    # might need to go outside
    if current_state[0] is not int:
        current_state = discretize(current_state)
    s = discretize(s)

    if done:
        Q[s[0], s[1], a] = current_reward
    if s[0]!=None:
        N[s[0], s[1], a]+= 1
        # update Q table 
        delta = alpha*(r + (gamma * np.max(Q[current_state[0], current_state[1]])) - Q[s[0], s[1],a])
        Q[s[0], s[1], a] += delta
    
    # use some exploration function to get next action
    a = epsilonGreedy(s)
    s = current_state
    r = current_reward

    return a


obs, _ = env.reset()
done = False
for step in range(EPISODES):
    i=0
    while not done:
        i+=1
        if i%100==0:
            print(i)
        action = QLearningAgent(obs, r, done)
        obs, r, done, truncated, info = env.step(action)

        env.render()
        time.sleep(0.001)
        
        # If the epsiode is up, then start another one
        if done:
            print(i)
            print("Episode {} done".format(EPISODES))
            env.reset()
        if truncated:
            print("Failed - start again")
            env.reset()
            
       
# Close the env
env.close()