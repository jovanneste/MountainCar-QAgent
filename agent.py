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

def randomAgent():
    return env.action_space.sample()


    

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