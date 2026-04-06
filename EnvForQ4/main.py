import argparse
import glob
from PIL import Image
from TreasureHunt.env import TreasureHunt,TreasureHunt_v2
from TreasureHunt.PolicyIteration import CPI

import numpy as np
import random
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

np.random.seed(12)
random.seed(12)

parser = argparse.ArgumentParser()
parser.add_argument('--solution', type=str, help='string that reflect question number and part, for example 1a')

###Map for actions
# UP = 0
# DOWN = 1
# LEFT = 2
# RIGHT = 3


def make_gif(src_folder, gif_name, duration=None):
    frames = [Image.open(image) for image in glob.glob(f"{src_folder}/*.png")]
    start_frame = frames[0]
    if duration=="None":
        duration=len(frames)//10
    start_frame.save(gif_name+".gif", format="GIF", append_images=frames, save_all=True, duration=duration, loop=0)






if __name__=="__main__":
    args = parser.parse_args()
    locations = {
                'ship': [(0,0)],
                'land': [(3,0),(3,1),(3,2),(4,2),(4,1),(5,2),(0,7),(0,8),(0,9),(1,7),(1,8),(2,7)],
                'fort': [(9,9)],
                'pirate': [(4,7),(8,5)],
                'treasure': [(4,0),(1,9)]
                }
    
    alphaList=[0.05]
    random.shuffle(alphaList)
    VList=[]
    for k in alphaList:
        env=TreasureHunt(locations)
        algo=CPI(env,k)
        pi,reward_array=algo.train()


        VList.append(reward_array)
    #     print(V_k)
        del algo
        env.visualize_policy(pi)
        env.visualize_policy_execution(pi,path=f'policy_iteration_{str(k)}.gif')


    print(VList)
    for i, V_k in enumerate(VList):
        plt.plot(np.arange(len(V_k)), V_k, label=f'alpha={alphaList[i]}')
    plt.legend() 
    plt.xlabel('Number of iterations')
    plt.ylabel('Evaluation reward')
    plt.title('CPI')
    plt.savefig('CPI.png')



