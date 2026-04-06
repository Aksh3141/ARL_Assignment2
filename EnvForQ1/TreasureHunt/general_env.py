import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pdb 
from grid import Grid 
from PIL import Image
from tqdm import tqdm
import torch
import imageio



UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class TreasureHunt(gym.Env):

    def __init__(self, locations, n = 7, is_testing = False):
        
        self.locations = locations
        self.is_testing = is_testing

        #the grid size
        self.n = n
        self.grid_dim = self.n * self.n
        self.num_treasures = len(self.locations['treasure'])
        self.num_states = self.n*self.n*(2**self.num_treasures)
        self.num_actions = 4
        self._action_delta = [[0,1],[0,-1],[-1,0],[1,0]]
        self.action_name = ['up','down','left','right']

        #the observation and action space
        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)

        #the 0th state
        self.initial_state = (2**self.num_treasures - 1) * self.grid_dim + 0
        self.state = self.initial_state

        #the treasure indicator
        self.treasure_from_index, self.index_from_treasure = self._get_treasure_indicator()

        #get transition matrix
        self.T = self._generate_tmatrix()

        #the reward matrix
        self.reward = self._generate_reward()
        self.number_of_steps = 0
        self.MAX_STEPS = 200 
    
    def reset(self):
        self.number_of_steps = 0
        if self.is_testing:
            self.state=self.initial_state
        else:
            self.state = np.random.randint(self.num_states-2)

        return self.state
    
    def locations_from_state(self, state):
        
        treasure_index = state // self.grid_dim
        treasure_indicator = self.treasure_from_index[treasure_index]
        treasure_locations = []
        for i in range(self.num_treasures):
            if(treasure_indicator[i] == '1'):
                treasure_locations.append(self.locations['treasure'][i])
        ship_location = state % self.grid_dim
        ship_location = (ship_location // self.n, ship_location % self.n)
        return ship_location, treasure_locations 

    def _get_treasure_indicator(self):
        
        treasure_indicator = []
        treasure_index = dict()
        for n in range(2**self.num_treasures):
    
            treasure_indicator_n = bin(n)[2:][::-1]
            treasure_indicator_n = treasure_indicator_n + ''.join(['0' for i in range(self.num_treasures - len(treasure_indicator_n))])
            treasure_indicator.append(treasure_indicator_n)

            treasure_index[treasure_indicator_n] = n 
        return treasure_indicator, treasure_index

    
    def _get_pos_ts(self, x, y):

        out = np.arange(0, 2**self.num_treasures)
        if((x,y) in self.locations['treasure']):
            ind = self.locations['treasure'].index((x,y))
            for i in range(2**self.num_treasures):
                tind = self.treasure_from_index[i]
                if(tind[ind] == '1'):
                    tind = list(tind)
                    tind[ind] = '0'
                    tind = ''.join(tind)
                    out[i] = self.index_from_treasure[tind]
        return out

    def step(self, action):
        self.number_of_steps += 1
        prev_treasure_config = self.state // self.grid_dim
        
        reward = self.reward[self.state]
        
        next_state_probs = self.T[self.state, action]
        self.state = np.random.multinomial(1, next_state_probs).nonzero()[0][0]

        curr_treasure_config = self.state // self.grid_dim
        treasure_obtained = (curr_treasure_config != prev_treasure_config)

        grid_position = self.state % self.grid_dim
        ship_location = (grid_position // self.n, grid_position % self.n)
        
        done = False
        if (ship_location == self.locations['fort'][0]):
            done = True 
        
        if (self.number_of_steps == self.MAX_STEPS):
            done = True 

        return self.state, reward, done, {'treasure_obtained': treasure_obtained}
    
    def is_land(self, x, y):
        return ((x,y) in self.locations['land'])

    def _get_grid_locations(self, state_id):

        locations = dict()
        treasure_index = state_id // (self.n*self.n)
        state_id = state_id % (self.n*self.n)

        x, y = state_id // self.n, state_id % self.n 

        locations['pirate'] = self.locations['pirate']
        locations['fort'] = self.locations['fort']
        locations['ship'] = [(x,y)]
        locations['land'] = self.locations['land']
        locations['treasure'] = []
        treasure_indicator = self.treasure_from_index[treasure_index]
        for i in range(self.num_treasures):
            if((x,y) == self.locations['treasure'][i]):
                continue
            if(treasure_indicator[i] == '0'):
                continue
            locations['treasure'].append(self.locations['treasure'][i])
        return locations

    def render(self, state_id = None, path = 'state.jpeg', return_image = True):
        if(state_id is None):
            state_id = self.state
        locations = self._get_grid_locations(state_id)
        grid = Grid(locations, N = self.n)
        if(return_image):
            image = grid.show(return_image = True)
            return image

    def visualize_policy_execution(self, policy, path = 'output.gif'):
        self.state = (2**self.num_treasures - 1)*self.n*self.n
 
        images = [self.render(return_image = True)]
        for i in tqdm(range(300)):
            
            action = policy.get_action(self.state)
            
            self.state, _, _, _ = self.step(action)

            images.append(self.render(return_image = True))
        
        pil_images = [Image.fromarray(arr.astype('uint8')) for arr in images]
        
        start_frame = pil_images[0]
        duration=2
        start_frame.save(path, format="GIF", append_images=pil_images, save_all=True, duration=duration, loop=0)

    def get_policy_rewards(self, policy):
        self.state = (2**self.num_treasures - 1)*self.n*self.n
        rewards = []
        for i in range(100):
            action = policy[self.state]
            self.state, reward = self.step(action)
            rewards.append(reward)
        
        return np.array(rewards)

    def visualize_neural_policy(self, policy, path='policy_vis.png'):
        n = self.n  
        locations = self.locations
        action_names = ['up', 'down', 'left', 'right']
        
        policy_grids = policy.get_policy_grid(n=n)
        
        for treasure_config in range(len(policy_grids)):
            policy_grid = policy_grids[treasure_config]
            
            state_id = treasure_config * n * n
            
            locations_vis = {
                'land': locations['land'],
                'fort': locations['fort'],
                'pirate': locations['pirate'],
                'treasure': []
            }
            
            treasure_indicator = bin(treasure_config)[2:].zfill(2)[::-1]
            for i in range(2):
                if treasure_indicator[i] == '1':
                    locations_vis['treasure'].append(locations['treasure'][i])
            
            for action_idx in range(4):
                arrow_positions = []
                for i in range(n):
                    for j in range(n):
                        if policy_grid[i, j] == action_idx:
                            arrow_positions.append((i, j))
                locations_vis[f'{action_names[action_idx]}_arrow'] = arrow_positions
            
            grid_vis = Grid(locations_vis, N=n)
            
            path_parts = path.split('.')
            path_with_config = '.'.join(path_parts[:-1] + [f'_{treasure_config}', path_parts[-1]])
            
            if treasure_config == 3:
                grid_vis.show(path_with_config)
                print(f"Saved policy visualization to: {path_with_config}")

    def visualize_policy(self, policy, path = 'policy_vis.png'):

        for i in range(2**self.num_treasures):
            state_id = i*self.n*self.n
            policy_i=[]
            for temp_i in range(state_id,state_id + self.n*self.n):
                action = policy.get_action(temp_i)
                policy_i.append(action)
            policy_i=np.array(policy_i)
            locations = self._get_grid_locations(state_id)
            del locations['ship']
            policy_i = policy_i.reshape(self.n, self.n)
            
            for j in range(4):
                policy_ij = (policy_i == j).nonzero()
                policy_ij = [(x,y) for x,y in zip(policy_ij[0], policy_ij[1])]
                locations[f'{self.action_name[j]}_arrow'] = policy_ij
            
            grid = Grid(locations, N = self.n)
            pathi = path.split('.')
            pathi = pathi[:-1] + [f"_{i}"] + [pathi[-1]]
            pathi = '.'.join(pathi)
            grid.show(pathi)


    def _generate_tmatrix(self):
        """
        If the agent is on land it will stay there.
        If the agent take an action, it will move in that direction with probability 0.9 if the move is valid else moves uniformly randomly
        to valid moves.
        With remaining probability it will move randomly to any of the valid states.
        Once treasure is collected it only moves to next treasure state such that treasures dont decrease. This means that if one
        treasure is collected, it will only move to keep the treasure count same or increase it. i.e 10-> 10 or 11
        """
        T = np.zeros((2**self.num_treasures, self.n, self.n, self.num_actions, 2**self.num_treasures, self.n, self.n))
        for x in range(self.n):
            for y in range(self.n):

                #if wall then ignore
                if(self.is_land(x,y)):
                    for i in range(2**self.num_treasures):
                        T[i,x,y,:,i,x,y] = 1
                    continue
                

                #it there is treasure at x,y then change the state
                ###the string keeps track of how many treasures have been collected
                pos_ts = self._get_pos_ts(x,y)

                #iterate over actions
                for a in range(4):

                    #the prob to distribute
                    prob = 1

                    #find the new state 
                    ####self._action_delta = [[0,1],[0,-1],[-1,0],[1,0]] [up, down, left, right]
                    nx = x + self._action_delta[a][0]
                    ny = y + self._action_delta[a][1]

                    #is valid?
                    if(nx < self.n and nx >= 0 and ny < self.n and ny >= 0):
                        if(not self.is_land(nx,ny)):
                            for i in range(2**self.num_treasures):
                                ###transition to new state with probability 0.9, do we need to iterate for all treasure states?
                                T[i,x,y,a,pos_ts[i],nx,ny] = 0.9
                            prob -= 0.9
                    

                    #now to distribute find all valid states
                    valid_states = [(x,y)]
                    for a_v in range(4):

                        #ignore the desired stated
                        if(a_v == a):
                            continue

                        nx = x + self._action_delta[a_v][0]
                        ny = y + self._action_delta[a_v][1]
                        #if valid append
                        if(nx < self.n and nx >= 0 and ny < self.n and ny >= 0):
                            if(self.is_land(nx,ny)):
                                continue
                            valid_states.append((nx, ny))
                    
                    #distribute the probability
                    for (nx,ny) in valid_states:
                        for i in range(2**self.num_treasures):
                            T[i,x,y,a,pos_ts[i],nx,ny] = prob / len(valid_states)

        T[:,self.n-1,self.n-1,:] = 0
        for i in range(2**self.num_treasures):
            T[i,self.n-1,self.n-1,:,i,self.n-1,self.n-1] = 1   

        T = T.reshape(self.num_states, 4, -1)
        return T
    
    def _generate_reward(self):

        reward = np.zeros((2**self.num_treasures,self.n,self.n)) - 0.1
        for i, tloc in enumerate(self.locations['treasure']):   
            for j, tind in enumerate(self.treasure_from_index):
                if(tind[i] == '1'):
                    reward[j,tloc[0],tloc[1]] = 2
    
        for ploc in self.locations['pirate']:
            reward[:,ploc[0],ploc[1]] = -1
        
        floc = self.locations['fort'][0]
        reward[:,floc[0], floc[1]] = 1.0
        return reward.reshape(-1)

# Locations for 10x10, n = 10
# locations1 = {
#     'ship': [(0,0)],
#     'land': [(3,0),(3,1),(3,2),(4,2),(4,1),(5,2),(0,7),(0,8),(0,9),(1,7),(1,8),(2,7)],
#     'fort': [(9,9)],
#     'pirate': [(4,7),(8,5)],
#     'treasure': [(4,0),(1,9)]
# }

# th = TreasureHunt(locations1)

# Locations for 7x7, n = 7
# locations2 = {
#     'ship': [(0, 0)],
#     'land': [
#         (2, 0), (2, 1), (3, 1), 
#         (0, 5), (0, 6), (1, 5)  
#     ],
#     'fort': [(6, 6)],
#     'pirate': [(3, 4), (5, 3)], 
#     'treasure': [(3, 0), (1, 6)] 
# }

# th = TreasureHunt(locations2)

                