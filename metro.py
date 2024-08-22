"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import random

import select_station
import wandb

import sys

from matplotlib import pyplot as plt

from metro_model import DRL4Metro, Encoder
import metro_vrp
from metro_vrp import MetroDataset

os_path = os.getcwd()
print(os_path)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# print(device)


class StateCritic(nn.Module): # ststic+ dynamic + vector present
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.preference_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum()
        return output

class StateCritic1(nn.Module): # ststic+ dynamic + matrix present
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic1, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.preference_encoder = Encoder(static_size, hidden_size)
        num_result, layers =1, 2


        # Define the encoder & decoder models
    #    self.fc1 = nn.Conv2d(hidden_size * 2, 20, kernel_size=5, stride=1, padding=2)
        if args.method_name == 'RLTD':
            num_result = 3
  
        self.fc1 = nn.Conv2d(hidden_size * layers, 128, kernel_size=5, stride=1, padding=2)
        self.fc2 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2)
        self.fc3 = nn.Linear(64 * args.grid_x_max * args.grid_y_max, num_result)

        #
        # self.fc3 = nn.Linear(20 * args.grid_x_max * args.grid_y_max, 36)
        # self.fc4 = nn.Linear(36, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic,hidden_size, grid_x_max, grid_y_max):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        
        dynamic_hidden = dynamic_hidden.view(hidden_size, grid_x_max, grid_y_max)

        static_hidden = static_hidden.view(hidden_size, grid_x_max, grid_y_max)
        hidden = torch.cat((static_hidden, dynamic_hidden), 0).unsqueeze(0)
        

        

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = output.view(output.size(0), -1)
        output = self.fc3(output)
        # output = self.fc4(output)
        return output


class Critic(nn.Module): # only dynamic0 + vector present

    def __init__(self, dynamic_size, hidden_size):
        super(Critic, self).__init__()

        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size, 64, kernel_size=1)
        self.fc2 = nn.Conv1d(64, 64, kernel_size=1)
        self.fc3 = nn.Conv1d(64, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, dynamic):

        dynamic_hidden = self.dynamic_encoder(dynamic)

        output = F.relu(self.fc1(dynamic_hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


class Critic1(nn.Module): # only dynamic0 + matrix present


    def __init__(self, dynamic_size, hidden_size):
        super(Critic1, self).__init__()

        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv2d(hidden_size, 20, kernel_size=5, stride=1, padding=2)
        self.fc2 = nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2)
        self.fc3 = nn.Linear(20 * args.grid_x_max * args.grid_y_max, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, dynamic, hidden_size, grid_x_max, grid_y_max):

        dynamic_hidden = self.dynamic_encoder(dynamic)

        dynamic_hidden = dynamic_hidden.view(hidden_size, grid_x_max, grid_y_max).unsqueeze(0)

        output = F.relu(self.fc1(dynamic_hidden))
        output = F.relu(self.fc2(output))
        output = output.view(output.size(0), -1)
        output = self.fc3(output)
        return output


def train(actor, critic, allowed_station, train_data, reward_fn,
         epoch_max, actor_lr, critic_lr, max_grad_norm, result_path, od_index_path, train_size, static_size, **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    
    #label = 'exp2_method_{}_initial_{}_budget_{}_w{}'.format(str(args.method_name), str(args.first), str(args.budget), str(args.w[0]))
    #label = 'exp3_method_{}_initial_{}_budget_{}'.format(str(args.method_name), str(args.first), str(args.budget))
    label = '{}_method_{}_w{}'.format(str(args.label_name), str(args.method_name),   str(args.w[0]))

    save_dir = os.path.join(result_path, label, now)


    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
    
    lam =0
    increment = 1/200

    average_reward_list, actor_loss_list, critic_loss_list, average_od_list, average_Ac_list, average_rad_list = [], [], [], [], [], []

    ideal = torch.tensor([100,100, 100]).unsqueeze(-1)
    

    # best_params = None
    best_reward = torch.tensor(0).to(device)
    best_reward_td = torch.tensor(100).to(device)

    dynamic = train_data.dynamic
 #   static = train_data.static
    static = metro_vrp.build_static(args.grid_x_max, args.grid_x_max, args.w, args.method_name, train_data.grid_num, static_size)
    grid_num = train_data.grid_num
    exist_line_num = train_data.exist_line_num
    line_full_tensor = train_data.line_full_tensor
    line_station_list = train_data.line_station_list
    inter_g = train_data.inter_g
    ExisInterList = train_data.ExisIndexList
    ExisLineStri = train_data.ExisLineStri

    od_matirx = metro_vrp.build_od_matrix1(grid_num, od_index_path)   #CPU need

    od_matirx =  od_matirx / torch.max(od_matirx)                     #GPU and CPU need
    # od_matirx = od_matirx.half()  # turn to float16 to recude CUDA memory  GPU need

    # exclude the needed od pair
    exclude_pair = metro_vrp.exlude_od_pair(args.grid_x_max)
    od_matirx = metro_vrp.od_matrix_exclude(od_matirx, exclude_pair)

    

    if args.social_equity:
    # path_house = r'/home/weiyu/program/metro_expand_combination/index_average_price.txt'
        price_matrix = metro_vrp.build_grid_price(args.path_house, args.grid_x_max, args.grid_y_max)
        price_matrix = price_matrix / torch.max(price_matrix)
        # price_matrix = price_matrix.half() # turn to float16 to recude CUDA memory : GPU

    if args.initial_direct:
        direction_list = args.initial_direct.split(',')
        initial_direct = []

        for i in direction_list:
            initial_direct.append(int(i))
    else:
        initial_direct = None

    if args.method_name == "GPI_PD":
        critics = [StateCritic1(5, 1, args.hidden_size).to(device) for _ in range(3)]
        [n_critic.train() for n_critic in critics]
        critics_opti = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in critics]
        

    for epoch in range(epoch_max):

        actor.train()
        critic.train()

        epoch_start = time.time()
        start = epoch_start

        od_list, social_equity_list, radiation_list = [], [], []
        for example_id in range(train_size):  # this loop accumulates a batch
            
            #once

            tour_idx, tour_logp = actor(static, dynamic, allowed_station,  args.station_num_lim, args.first, args.budget, args.initial_direct, args.line_unit_price, args.station_price, decoder_input=None, last_hh=None)

            tour_idx_cpu = tour_idx.cpu()
            tour_idx_np = tour_idx_cpu.numpy()
            agent_grid_list = tour_idx_np[0].tolist()

            reward_od = metro_vrp.reward_fn1(tour_idx_cpu, grid_num, agent_grid_list, line_full_tensor, line_station_list,
                                              exist_line_num, od_matirx, args.grid_x_max, args.dis_lim)  #CPU
            agent_Ac = metro_vrp.agent_grids_price(tour_idx_cpu, args.grid_x_max, price_matrix) #cpu

            radiation_ac = metro_vrp.radiation_ac(tour_idx_cpu, grid_num, agent_grid_list, line_full_tensor, line_station_list,
                                              exist_line_num, od_matirx, args.grid_x_max, args.dis_lim,  ExisInterList, inter_g, ExisLineStri)

            reward_vec = torch.Tensor(np.array([reward_od, agent_Ac, radiation_ac]).transpose()).unsqueeze(-1)
            
            od_list.append(reward_od.item())
            social_equity_list.append(agent_Ac.item())
            radiation_list.append(radiation_ac.item())



            if args.social_equity == 1:
                reward =torch.einsum("nb,bn->n", torch.Tensor(args.w), reward_vec )
                reward = reward.to(device)


            elif args.social_equity == 2:

                od_list.append(0)
                social_equity_list.append(agent_Ac.item())

                reward = args.factor_weight1*reward_od + (1 - args.factor_weight1)*agent_Ac

                # reward = args.factor_weight1 * reward_od - 0.5 * agent_Ac
                reward = reward.to(device)

            else:
                reward = metro_vrp.reward_fn1(tour_idx_cpu, grid_num, agent_grid_list, line_full_tensor, line_station_list,
                                      exist_line_num, od_matirx, args.grid_x_max, args.dis_lim)

                od_list.append(reward.item())
                social_equity_list.append(0)
                reward = reward.to(device)



            # Query the critic for an estimate of the reward

            # critic_est = critic(static, dynamic).view(-1)   # ststic+ dynamic + vector present

            if args.method_name != "GPI_PD":

                critic_est = critic(static, dynamic, args.hidden_size, args.grid_x_max, args.grid_y_max).view(-1)  # ststic+ dynamic + matrix present

            if args.method_name == 'RLTD':
                diff1 =torch.einsum("nb,bn->nb", args.w,  ideal-reward_vec).to(device)

                diff1_max, a = torch.max(diff1,1)

                diff2 =torch.einsum("nb,bn->nb", args.w,  ideal- critic_est)
            

                diff2_max, a = torch.max(diff2,1)
 
                advantage = diff2_max - diff1_max

                la = torch.dist(reward_vec.to(device),critic_est,p=2)
          
                cristic_r = torch.sum(args.w * critic_est)

                lb = (reward - cristic_r)
                
                critic_adv = (1-lam) * lb + lam * la

            elif args.method_name == 'GPI_PD':

                critics_est = [new_critic(static, dynamic, args.hidden_size, args.grid_x_max, args.grid_y_max).view(-1) for new_critic in critics]
                advantage = reward - torch.max(torch.tensor(critics_est))


            elif args.method_name == 'RLEU':
                new_w = np.random.dirichlet(np.ones(3),1)
                new_static = metro_vrp.build_static(args.grid_x_max, args.grid_x_max, new_w, args.method_name, train_data.grid_num, static_size)
                tour_idx, tour_logp = actor(new_static, dynamic, allowed_station, args.station_num_lim, args.first, args.budget, initial_direct, args.line_unit_price, args.station_price, decoder_input=None, last_hh=None)

                tour_idx_cpu = tour_idx.cpu()
                tour_idx_np = tour_idx_cpu.numpy()
                agent_grid_list = tour_idx_np[0].tolist()

                reward_od = metro_vrp.reward_fn1(tour_idx_cpu, grid_num, agent_grid_list, line_full_tensor, line_station_list,
                                                exist_line_num, od_matirx, args.grid_x_max, args.dis_lim)  #CPU
                agent_Ac = metro_vrp.agent_grids_price(tour_idx_cpu, args.grid_x_max, price_matrix) #cpu

                radiation_ac = metro_vrp.radiation_ac(tour_idx_cpu, grid_num, agent_grid_list, line_full_tensor, line_station_list,
                                                exist_line_num, od_matirx, args.grid_x_max, args.dis_lim,  ExisInterList, inter_g, ExisLineStri)

                new_reward_vec = torch.Tensor(np.array([reward_od, agent_Ac, radiation_ac]).transpose()).unsqueeze(-1)
                
                n_reward =torch.einsum("nb,bn->n", args.w, new_reward_vec )
                n_reward = n_reward.to(device)
                adv_reward = n_reward if n_reward > reward else reward
                advantage = (adv_reward -critic_est)
            else:
                advantage = (reward - critic_est)

            per_actor_loss = -advantage.detach()*tour_logp.sum(dim=1)  #loss
            if args.method_name == 'RLTD':
                per_critic_loss = critic_adv **2
            else:
                per_critic_loss = advantage ** 2

            if example_id == 0:
                actor_loss0 = per_actor_loss
                critic_loss0 = per_critic_loss
                rewards = reward
                if args.method_name == 'RLTD':
                    max_differ = diff1_max
            else:
                actor_loss0 = actor_loss0 + per_actor_loss
                critic_loss0 = critic_loss0 + per_critic_loss  #batch culmulative
                rewards = rewards + reward
                if args.method_name == 'RLTD':
                    max_differ += diff1_max

        

        actor_loss = actor_loss0 / train_size
        critic_loss = critic_loss0 / train_size
        average_reward = rewards / train_size
        average_differ = max_differ / train_size
        average_od = sum(od_list)/len(od_list) #avg od reward
        average_Ac = sum(social_equity_list)/len(social_equity_list) #avg equity reward
        average_rad = sum(radiation_list)/len(social_equity_list)
        lam += increment


        if args.log :
                wandb.log(
                    {
                        "reward/satified OD demand": average_od,
                        "reward/social equity": average_Ac,
                        "reward/radiation accessibility": average_rad,
                        "loss/critic_loss": critic_loss.item(),
                        "loss/actor_loss": actor_loss.item(),
                    })
    
        average_reward_list.append(average_reward.half().item())



        actor_loss_list.append(actor_loss.half().item())
        critic_loss_list.append(critic_loss.half().item())
        average_od_list.append(average_od)
        average_Ac_list.append(average_Ac)
        average_rad_list.append(average_rad)

        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        actor_optim.step()

        if args.method_name != "GPI_PD":
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()
        else:
            critic_loss.requires_grad_(True)
            for critic0_opti, critic0 in zip(critics_opti, critics):
                critic0_opti.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic0.parameters(), max_grad_norm)
                critic0_opti.step()

        end = time.time()
        cost_time = end - start

        print('epoch %d,  actor_loss: %2.4f,  critic_loss: %2.4f, cost_time: %2.4f, best_solution: %s,od: %2.4f, eqity: %2.4f,rad: %2.4f'
              % (epoch,  actor_loss.item(), critic_loss.item(), cost_time, best_reward.tolist(),average_od, average_Ac, average_rad))

        torch.cuda.empty_cache() # reduce memory


        average_reward_value = average_reward.item() 
        
        if  args.method_name == 'RLTD':
            if best_reward_td > average_differ:
                print(best_reward_td, average_differ)
                best_reward_td = average_differ
                save_path = os.path.join(save_dir, 'actor.pt')
                torch.save(actor.state_dict(), save_path)

                save_path = os.path.join(save_dir, 'critic.pt')
                torch.save(critic.state_dict(), save_path)
                epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
                if not os.path.exists(epoch_dir):
                   os.makedirs(epoch_dir)
                save_path = os.path.join(epoch_dir, 'actor.pt')
                torch.save(actor.state_dict(), save_path)

                save_path = os.path.join(epoch_dir, 'critic.pt')
                torch.save(critic.state_dict(), save_path)

                best_od, best_ac, best_ra = average_od, average_Ac,average_rad

                

                

        elif average_reward_value > best_reward:
            best_reward = average_reward

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)
            epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
            if not os.path.exists(epoch_dir):
                   os.makedirs(epoch_dir)
            save_path = os.path.join(epoch_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(epoch_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

            best_od, best_ac, best_ra = average_od, average_Ac,average_rad

            print(best_od,best_ac,best_ra)




    records_path = os.path.join(save_dir, 'reward_actloss_criloss.txt')
    best_records_path = os.path.join(result_path,  'exp5_method_{}_ini_{}_best_record.txt'.format(str(args.label_name), str(args.first)))

    write_file = open(records_path, 'w')
    best_file = open(best_records_path,'a')
    best_write = str(best_od) +'\t' + str(best_ac) +'\t' + str(best_ra) + '\n'
    best_file.write(best_write)
    best_file.close()


    for i in range(epoch_max):
        per_average_reward_record = average_reward_list[i]
        per__actor_loss_record = actor_loss_list[i]
        per_critic_loss_record = critic_loss_list[i]
        per_epoch_od = average_od_list[i]
        per_epoch_Ac = average_Ac_list[i]
        per_rad_Ac = average_rad_list[i]

        to_write = str(per_average_reward_record) +'\t' + str(per__actor_loss_record) + '\t'+ str(per_critic_loss_record) + '\t'+ str(per_epoch_od) + '\t' + str(per_epoch_Ac) + '\t'+str(per_rad_Ac) + '\n'

        write_file.write(to_write)
    write_file.close()





def train_vrp(args):


    STATIC_SIZE = 2 + 3
    DYNAMIC_SIZE = 1

    if args.method_name == 'RLSM' :
        STATIC_SIZE = 2

    #generage a random preference
    
    #args.w = torch.Tensor(np.array(args.w)).to(device).unsqueeze(-1).unsqueeze(0)
    args.w = np.array([args.w])

    print(args.w)

    if args.log:

        wandb.init(project='metro',
			name=args.label_name +args.method_name +"_"+str(args.w[0][0]) + "_"+str(args.budget),
			config=args)

    allowed_station = select_station.initial_final_station(args.first)



    train_data = MetroDataset(args.grid_x_max, args.grid_y_max, args.w, args.method_name, args.exist_line_num,  args.initial_station,  STATIC_SIZE, DYNAMIC_SIZE)

    actor = DRL4Metro(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    train_data.v_to_g,
                    train_data.vector_allow,
                    args.num_layers,
                    args.dropout).to(device)


    # critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)  # ststic+ dynamic0 + vector present

    critic = StateCritic1(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)


    # critic = Critic(DYNAMIC_SIZE, args.hidden_size).to(device)                    # only dynamic0 + vector present

    # critic = Critic1(DYNAMIC_SIZE, args.hidden_size).to(device)                   # only dynamic0 + matrix present

    kwargs = vars(args)  # dict

    kwargs['train_data'] = train_data
    kwargs['reward_fn'] = metro_vrp.reward_fn



    if args.checkpoint:  # test: give model_solution  
        static, dynamic= train_data.static, train_data.dynamic
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

        tour_idx, tour_logp   = actor(static, dynamic, allowed_station,  args.station_num_lim, args.first, args.budget, args.initial_direct, args.line_unit_price, args.station_price, decoder_input=None, last_hh=None)

        print(tour_idx)


      #  tour_idx, tour_logp, dynamic0 = actor(static, dynamic, decoder_input=None, last_hh=None)

        now = '%s' % datetime.datetime.now().time()
        now = now.replace(':', '_')


        result_path = os.path.join(args.result_path,  'new_station/exp5_method_{}_{}_w{}'.format(str(args.method_name), str(args.label_name),str(args.w)))

        now = '%s' % datetime.datetime.now().time()
        now = now.replace(':', '_')



        if not os.path.exists(result_path):
            os.makedirs(result_path)

        model_solution_path = os.path.join(result_path,  'tour_idx.txt')

        
        

      #  f = open(model_solution_path, 'w')
        f = open(model_solution_path, 'w')

        to_write = ''
        for i in tour_idx[0]:
            to_write = to_write + str(i.item()) + ','

        to_write1 = to_write.rstrip(',')
        f.write(to_write1)
        f.close()



    if not args.test:  # train
        train(actor, critic, allowed_station, static_size=STATIC_SIZE, **kwargs)







def model_solution(grid_x_max, grid_y_max, exist_line_num, hidden_size,path, initial_station, initial_direct,station_num_lim,
 budget, line_unit_price, station_price):
    # grid_x_max, grid_y_max = 29, 29
    # hidden_size =
    # path = r'/home/weiyu/program/metro_expand_combination/result/21_04_10.890831/actor.pt'

    from metro_vrp import MetroDataset

    STATIC_SIZE = 2 + 3
    DYNAMIC_SIZE = 1

    train_data = MetroDataset(grid_x_max, grid_y_max, exist_line_num, initial_station, static_size=2, dynamic_size=1)

    static = train_data.static
    dynamic = train_data.dynamic

    model = DRL4Metro(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    train_data.v_to_g,
                    train_data.vector_allow,
                    num_layers=1,
                    dropout=0.).to(device)


    model.load_state_dict(torch.load(path))
    model.eval()   
    # model.train() 

    tour_idx, tour_logp = model(static, dynamic, station_num_lim, budget,
                                initial_direct, line_unit_price, station_price, decoder_input=None, last_hh=None)

    print('tour_idx:', tour_idx)


def set_seed(seed):
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子

    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
     a=np.random.dirichlet(np.ones(3),size=1)

     parser = argparse.ArgumentParser(description='Combinatorial Optimization')
     parser.add_argument('--grid_x_max', default=29, type=int)
     parser.add_argument('--grid_y_max', default=29, type=int)
    
     parser.add_argument('--checkpoint',  default=None)
     
    
     parser.add_argument('--test', action='store_true', default=False)
    
     parser.add_argument('--actor_lr', default=5e-4, type=float)
     parser.add_argument('--critic_lr', default=5e-4, type=float)
     parser.add_argument('--max_grad_norm', default=2, type=float)
     parser.add_argument('--batch_size', default=16, type=int)
     parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
     parser.add_argument('--dropout', default=0.1, type=float)
     parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    
     parser.add_argument('--train_size',default=256, type=int)   # similar to batch size
    
     parser.add_argument('--exist_line_num', default=2, type=int)
     parser.add_argument('--epoch_max', default=4000, type=int) # the number of total epoch
    
     parser.add_argument('--initial_station', default= 0, type=int)
    # default=None when there is no initial direct, otherwise example: default ='[812, 783, 784],[812, 813, 783, 784, 754, 755], [0, 1, 2, 29, 30, 31]', type = str
     parser.add_argument('--first', type=int, default=None, help='for ddid ')

     parser.add_argument('--initial_direct', default=None)
     # default=None when there is no initial direct, otherwise example: default ='0,2', type = str
    
     parser.add_argument('--station_num_lim', default=45, type=int)  # limit the number of stations in a line
     parser.add_argument('--budget', default=270, type=int)


     parser.add_argument('--line_unit_price', default=1.0 , type=float)
     parser.add_argument('--station_price', default=5.0, type=float)
    
     parser.add_argument('--dis_lim', default=None)


     parser.add_argument('--log', default=True, type=bool)
       
     parser.add_argument('--method_name', default='RLTD', metavar='ENVNAME',
                    help='environment to train on: RLWS | RLEU | RLSM | RLTD |GPI_PD|GPI_LS')
     
     parser.add_argument('--label_name', default='exp5_dyna_', metavar='ENVNAME',
                    help='')

     parser.add_argument('--w', type=float, nargs='+',default=[0.1,0.5,0.4], help='for ddid ')
    
     parser.add_argument('--social_equity', default=1, type=int)
     parser.add_argument('--factor_weight', default=1, type=float)
     parser.add_argument('--factor_weight1', default=0, type=float)

     
     parser.add_argument('--result_path', default= os_path + '/result/', type=str)
     parser.add_argument('--od_index_path', default= os_path + '/od_index_dyna.txt', type=str)
     parser.add_argument('--path_house', default= os_path + '/index_average_price_dyna.txt', type=str)
    
     args = parser.parse_args()

  
     train_vrp(args)











