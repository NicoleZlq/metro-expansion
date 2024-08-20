


import os
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely import box, LineString, normalize, Polygon
import shapely
from scipy.ndimage import gaussian_filter1d


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MetroDataset(Dataset):
    def __init__(self, grid_x_max, grid_y_max, weight, method, exist_line_num, initial_station = None, static_size=2, dynamic_size=1):
       

        super(MetroDataset, self).__init__()

        self.grid_x_max = grid_x_max
        self.grid_y_max = grid_y_max
        self.grid_index1_max = grid_y_max - 1 # The max index of the first dimension
        self.grid_index2_max = grid_x_max - 1 # The max index of the second dimension
        self.static_size = static_size
        self.dynamic_size = dynamic_size

        self.grid_num = grid_x_max*grid_y_max
        self.exist_line_num = exist_line_num
        self.method = method


        self.positive = torch.Tensor([1]).long().to(device)
        self.negative = torch.Tensor([-1]).long().to(device)
        self.only1 = torch.tensor([1]).to(device)
        self.sign = torch.Tensor([-1]).long().to(device)
        
        self.quadrant1 = torch.tensor([-1, 1]).view(1, 2).to(device) 
        self.quadrant2 = torch.tensor([-1, -1]).view(1, 2).to(device)  
        self.quadrant3 = torch.tensor([1, -1]).view(1, 2).to(device) 
        self.quadrant4 = torch.tensor([1, 1]).view(1, 2).to(device)  
        self.quadrant_up = torch.tensor([-1, 0]).view(1, 2).to(device)  
        self.quadrant_right = torch.tensor([0, 1]).view(1, 2).to(device)  
        self.quadrant_down = torch.tensor([1, 0]).view(1, 2).to(device)  
        self.quadrant_left = torch.tensor([0, -1]).view(1, 2).to(device)  

        a = []
        self.null_tensor = torch.Tensor(a).long().to(device)  #tensor([], device='cuda:0', dtype=torch.int64) torch.Size([0])


        # build dynamic
        self.dynamic = torch.zeros((1, self.dynamic_size, self.grid_num)).float().to(device)# size with batch

        


        # #build static
        # #RLSM dont have the weight information
        def build_static(grid_x_max, grid_y_max, weight):
            
          
          for i in range(grid_y_max):
              per_need = np.zeros((grid_x_max, 2))
              
              for j in range(grid_x_max):
                  per_need[j, 0] = i
                  per_need[j, 1] = j

              if i == 0:
                  need = per_need
              else:
                  need = np.vstack((need, per_need))

          np_static = need.transpose()
          weight = weight.transpose()
          
          if self.method != 'RLSM':
            weight_need = np.full([3,grid_x_max * grid_y_max], weight) # 3,841
            np_static = np.vstack((np_static,weight_need)) #5,841
          return np_static
        

        np_static = build_static(grid_x_max, grid_y_max, weight)
        ten_static = torch.from_numpy(np_static).float()
        self.static = ten_static.view(1, self.static_size, self.grid_num).to(device) # size with batch


 ########## build existing lines
        def g_to_v1(agent_grids):

            vector_index = agent_grids[:, 0] * grid_x_max + agent_grids[:, 1]

            return vector_index

        def process_line(index_line, exi_sta_adj_sta):
            
            for i in range(len(index_line)):
                this_index = index_line[i]
                if this_index not in exi_sta_adj_sta:
                    exi_sta_adj_sta[this_index] = []

                if (i - 2) >= 0:
                    exi_sta_adj_sta[this_index].append(index_line[i - 2])
                if (i - 1) >= 0:
                    exi_sta_adj_sta[this_index].append(index_line[i - 1])
                if (i + 1) <= (len(index_line) - 1):
                    exi_sta_adj_sta[this_index].append(index_line[i + 1])
                if (i + 2) <= (len(index_line) - 1):
                    exi_sta_adj_sta[this_index].append(index_line[i + 2])
#########################################################################################################
        # original representation of each line
        line0_ststion_list = [[8, 2], [10, 3], [10, 5], [11, 6], [11, 7], [12, 9], [12, 11], [12, 12], [12, 13],
                              [12, 14], [12, 15], [12, 16], [12, 17], [12, 18], [12, 20], [11, 22], [11, 23], [11, 24],
                              [11, 25]]
        line1_ststion_list = [[0, 13], [1, 14], [3, 14], [4, 14], [5, 14], [6, 14], [8, 14], [9, 14], [10, 14],
                              [12, 14], [13, 14], [14, 14], [15, 14], [16, 14], [17, 14], [18, 14], [20, 14], [21, 14],
                              [23, 14], [24, 14], [26, 14]]
        

        line2_ststion_list = [[3, 26], [5, 24], [6, 22], [6, 21], [7, 21], [7, 20], [7, 19], [8, 19], [9, 19], [9, 18], [10, 18], [11, 18], [11, 17], [12, 17], [13, 17], [13, 16], 
                           [13,15], [13, 14], [13, 13], [14, 13], [15, 13], [15, 12], [16, 12], [16, 11], [17, 11], [17, 10], [17, 9], [18, 9], [19, 9], [19, 8], [19, 7], [20, 7], [21, 7], [22, 7], [22, 6]]


        


        line3_station_list = [[5, 0], [7, 0], [8, 1], [8, 2], [9, 2], [9, 3], [9, 4], [10, 4], [11, 4], [11, 5], [11, 6], [12, 6], [12, 7], [13, 7], [13, 8], [14, 8], [14, 9], [14, 10], [15, 10], [15, 11], [16, 11], [17, 12], [17, 13], [17, 14], [17, 15], [18, 15], [18, 16], [19, 16], [19, 17], [20, 17], [20, 18], [20, 19], [21, 19], [21, 20], [22, 20]]

        ########################## the first step for add new lines, the total need 4 steps
        # need used the below function exlude_od_pair(grid_x_max) to exclude the od pair alone added line


        # att3_5
        # index_line4_station = [815, 759, 730, 701, 672, 644, 615, 587, 559, 531, 503, 504, 476, 477, 448, 419, 391, 392, 393, 365,
        #                        337, 308, 279, 250, 222, 223, 224,225, 197, 168, 140, 112]

        self.existing_line = [line0_ststion_list, line1_ststion_list]

        #added new line

        self.existing_line.append(line2_ststion_list)

        self.existing_line.append(line3_station_list)



        ##########################
        np_line0_station = np.array(line0_ststion_list)
        np_line1_station = np.array(line1_ststion_list)


        index_line0_station = g_to_v1(np_line0_station)
        index_line1_station = g_to_v1(np_line1_station)


        

        index_line0_station = [int(i) for i in index_line0_station]
        index_line1_station = [int(i) for i in index_line1_station]

        #added new line

        


        index_line_station_list = []
        index_line_station_list.append(index_line0_station)
        index_line_station_list.append(index_line1_station)


        #added new line
        np_line2_station = np.array(line2_ststion_list)
        np_line3_station = np.array(line3_station_list)
        index_line2_station = g_to_v1(np_line2_station)
        index_line3_station = g_to_v1(np_line3_station)

        index_line2_station = [int(i) for i in index_line2_station]
        index_line3_station = [int(i) for i in index_line3_station]
        index_line_station_list.append(index_line2_station)
        index_line_station_list.append(index_line3_station)


        self.line_station_list = index_line_station_list

##################build full cross grid including the grids which have no station

        line0_full_list = [[8, 2], [10, 3], [10, 4], [10, 5], [11, 6], [11, 7], [12, 9], [12, 10],[12, 11], [12, 12], [12, 13],
                           [12, 14], [12, 15], [12, 16], [12, 17], [12, 18], [12, 19], [12, 20], [11, 22], [11, 23], [11, 24],
                           [11, 25]]

        line1_full_list = [[0, 13], [1, 14], [2, 14], [3, 14], [4, 14], [5, 14], [6, 14],[7, 14], [8, 14], [9, 14], [10, 14],[11, 14],
                           [12, 14], [13, 14], [14, 14], [15, 14], [16, 14], [17, 14], [18, 14],[19, 14], [20, 14], [21, 14],[22, 14],
                           [23, 14], [24, 14],[25, 14], [26, 14]]
        line2_full_list = [[3, 26], [5, 24], [6, 22], [6, 21], [7, 21], [7, 20], [7, 19], [8, 19], [9, 19], [9, 18], [10, 18], [11, 18], [11, 17], [12, 17], [13, 17], [13, 16], 
                           [13,15], [13, 14], [13, 13], [14, 13], [15, 13], [15, 12], [16, 12], [16, 11], [17, 11], [17, 10], [17, 9], [18, 9], [19, 9], [19, 8], [19, 7], [20, 7], [21, 7], [22, 7], [22, 6]]
        
        line3_full_list =  [[5, 0], [7, 0], [8, 1], [8, 2], [9, 2], [9, 3], [9, 4], [10, 4], [11, 4], [11, 5], [11, 6], [12, 6], [12, 7], [13, 7], [13, 8], [14, 8], [14, 9], 
                               [14, 10], [15, 10], [15, 11], [16, 11], [17, 12], [17, 13], [17, 14], [17, 15], [18, 15], [18, 16], [19, 16], [19, 17], [20, 17], [20, 18], [20, 19], [21, 19], [21, 20], [22, 20]]
        
        #
        # line2_full_list = [[15, 5], [15, 6], [15, 7], [15, 8], [15, 9], [16, 10], [17, 11], [17, 12], [17, 13], [17, 14], [17, 15], [17, 16], [17, 17],
        #                    [16, 18], [15, 19], [14, 19], [13, 19], [11, 18], [10, 18], [9, 18], [8, 18], [7, 20], [7, 21], [7, 22], [7, 23],
        #                    [5, 24], [3, 23], [2, 22], [0, 23]]
        #
        # line3_full_list = [[0, 13], [1, 12], [2, 12], [3, 12], [4, 13], [4, 14], [4, 15], [4, 16], [5, 16], [6, 16],
        #                    [7, 16], [8, 16], [9, 16], [10, 16], [11, 16], [12, 15], [13, 15], [14, 15], [15, 16], [16, 16], [17, 16],
        #                    [18, 16], [20, 17], [21, 17], [22, 17], [23, 17], [24, 17], [25, 17], [26, 17], [26, 18], [26, 19], [26, 20]]

        np_line0_full = np.array(line0_full_list)
        np_line1_full = np.array(line1_full_list)
        

        index_line0_full = g_to_v1(np_line0_full)
        index_line1_full = g_to_v1(np_line1_full)
        

        index_line0_full = [int(i) for i in index_line0_full]
        index_line1_full = [int(i) for i in index_line1_full]
        

        index_line_full_list = []
        index_line_full_list.append(index_line0_full)
        index_line_full_list.append(index_line1_full)
        

      #added new line
        np_line2_full = np.array(line2_full_list)
        np_line3_full = np.array(line3_full_list)
        index_line2_full = g_to_v1(np_line2_full)
        index_line3_full = g_to_v1(np_line3_full)
        index_line2_full = [int(i) for i in index_line2_full]
        index_line3_full = [int(i) for i in index_line3_full]
        index_line_full_list.append(index_line2_full)
        index_line_full_list.append(index_line3_full)


        index_line_full_list.append(index_line3_full)

       
        ###
        exi_sta_adj_sta = {}
        index_line_list = index_line_full_list

        for j in index_line_list:
            process_line(j, exi_sta_adj_sta)

        # qu chong
        for key, value in exi_sta_adj_sta.items():
            value = list(set(value))
            exi_sta_adj_sta[key] = value

        self.exi_sta_adj_sta = exi_sta_adj_sta

        # line_full_tensor
        self.line_full_tensor = []
###############
        # CPU
        line_full_tensor0 = torch.tensor(index_line0_full).view(len(line0_full_list), 1)
        line_full_tensor1 = torch.tensor(index_line1_full).view(len(line1_full_list), 1)
        

        self.line_full_tensor.append(line_full_tensor0)
        self.line_full_tensor.append(line_full_tensor1)

        #added new line
        line_full_tensor2 = torch.tensor(index_line2_full).view(len(line2_full_list), 1)
        line_full_tensor3 = torch.tensor(index_line3_full).view(len(line3_full_list), 1)
        self.line_full_tensor.append(line_full_tensor2)
        self.line_full_tensor.append(line_full_tensor3)

        first_line1, first_line2 = [i[0] for i in self.existing_line[0]], [i[1] for i in self.existing_line[0]]
        second_line1, second_line2 = [i[0] for i in self.existing_line[1]], [i[1] for i in self.existing_line[1]]

        first_line = LineString(np.column_stack((first_line1, first_line2)))

        second_line = LineString(np.column_stack((second_line1, second_line2)))

        ExisLineStri = [first_line, second_line]

        Inter = np.array((shapely.intersection(second_line, first_line)).xy)

        inter_g = int(Inter[0]*grid_x_max + Inter[1])


        ExisIndexList =[index_line_station_list[0][:index_line_station_list[0].index(inter_g)], index_line_station_list[1][:index_line_station_list[1].index(inter_g)]]


        self.ExisIndexList =  ExisIndexList
        self.inter_g = inter_g
        self.ExisLineStri = ExisLineStri

        # self.line_full_tensor.append(line_full_tensor2)
        # self.line_full_tensor.append(line_full_tensor3)

        #################### the fourth step for add new lines, the total need 4 steps
        # line_full_tensor4 = torch.tensor(index_line4_station).view(len(index_line4_station), 1)
        # self.line_full_tensor.append(line_full_tensor4)
        # #
        # line_full_tensor5 = torch.tensor(index_line5_station).view(len(index_line5_station), 1)
        # self.line_full_tensor.append(line_full_tensor5)
        # # #
        # line_full_tensor6 = torch.tensor(index_line6_station).view(len(index_line6_station), 1)
        # self.line_full_tensor.append(line_full_tensor6)
        #
        # line_full_tensor7 = torch.tensor(index_line7_station).view(len(index_line7_station), 1)
        # self.line_full_tensor.append(line_full_tensor7)
        #
        # line_full_tensor8 = torch.tensor(index_line8_station).view(len(index_line8_station), 1)
        # self.line_full_tensor.append(line_full_tensor8)
#####################
        # GPU
        # line_full_tensor0 = torch.tensor(index_line0_full).view(len(line0_full_list), 1).to(device)
        # line_full_tensor1 = torch.tensor(index_line1_full).view(len(line1_full_list), 1).to(device)
        # line_full_tensor2 = torch.tensor(index_line2_full).view(len(line2_full_list), 1).to(device)
        # line_full_tensor3 = torch.tensor(index_line3_full).view(len(line3_full_list), 1).to(device)
        #
        # self.line_full_tensor.append(line_full_tensor0)
        # self.line_full_tensor.append(line_full_tensor1)
        # self.line_full_tensor.append(line_full_tensor2)
        # self.line_full_tensor.append(line_full_tensor3)
        #
        # #################### the fourth step for add new lines, the total need 4 steps
        # line_full_tensor4 = torch.tensor(index_line4_station).view(len(index_line4_station), 1).to(device)
        # self.line_full_tensor.append(line_full_tensor4)
        #
        # line_full_tensor5 = torch.tensor(index_line5_station).view(len(index_line5_station), 1).to(device)
        # self.line_full_tensor.append(line_full_tensor5)
################################

############################################################################################################

        def increment_1():
            #output-- grid_inc_1: CUDA size (9,2)
            grid_list = [[-2,0],[-2,1],[-2,2],[-1,0],[-1,1],[-1,2],[0,1],[0,2]]
            #grid_list = [[-2,1],[-2,2],[-1,1],[-1,2]]

            grid_inc_1 = torch.tensor(grid_list).to(device)

            return grid_inc_1  # CUDA  torch.Size([8, 2]) dtype: torch.int64

        def increment_2():
            
            #output-- grid_inc_2: CUDA size (9,2)
            grid_list = [[-2,-2],[-2,-1],[-2,0],[-1,-2],[-1,-1],[-1,0],[0,-2],[0,-1]]

            #grid_list = [[-2,-2],[-2,-1],[-1,-2],[-1,-1]]

            grid_inc_2 = torch.tensor(grid_list).to(device)

            return grid_inc_2  # CUDA  torch.Size([8, 2]) dtype: torch.int64

        def increment_3():
            
            #output-- grid_inc_2: CUDA size (9,2)
            grid_list = [[0,-2],[0,-1],[1,-2],[1,-1],[1,0],[2,-2],[2,-1],[2,0]]

            #grid_list = [[1,-2],[1,-1],[2,-2],[2,-1]]

            grid_inc_3 = torch.tensor(grid_list).to(device)

            return grid_inc_3  # CUDA  torch.Size([8, 2]) dtype: torch.int64

        def increment_4():
            
            #output-- grid_inc_2: CUDA size (9,2)
            grid_list = [[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]

            #grid_list = [[1,1],[1,2],[2,1],[2,2]]

            grid_inc_4 = torch.tensor(grid_list).to(device)

            return grid_inc_4  # CUDA  torch.Size([8, 2]) dtype: torch.int64

        def increment_up():
            
            #output-- grid_inc_up: CUDA size (9,2)
            #grid_list = [[-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],[0,-2],[0,-1],[0,1],[0,2]]

            grid_list = [[-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2]]

            grid_inc_up = torch.tensor(grid_list).to(device)

            return grid_inc_up  # CUDA  torch.Size([14, 2]) dtype: torch.int64

        def increment_right():
            
            #output-- grid_inc_up: CUDA size (9,2)
            #grid_list = [[-2,0],[-2,1],[-2,2],[-1,0],[-1,1],[-1,2],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
            grid_list = [[-2,1],[-2,2],[-1,1],[-1,2],[0,1],[0,2],[1,1],[1,2],[2,1],[2,2]]

            grid_inc_right = torch.tensor(grid_list).to(device)

            return grid_inc_right  # CUDA  torch.Size([14, 2]) dtype: torch.int64

        def increment_down():
            
            #output-- grid_inc_up: CUDA size (9,2)
            #grid_list = [[0,-2],[0,-1],[0,1],[0,2],[1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-2],[2,-1],[2,0],[2,1],[2,2]]
            grid_list = [[1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-2],[2,-1],[2,0],[2,1],[2,2]]

            grid_inc_down = torch.tensor(grid_list).to(device)

            return grid_inc_down # CUDA  torch.Size([14, 2]) dtype: torch.int64

        def increment_left():
            
            #output-- grid_inc_up: CUDA size (9,2)
            #grid_list = [[-2,-2],[-2,-1],[-2,0],[-1,-2],[-1,-1],[-1,0],[0,-2],[0,-1],[1,-2],[1,-1],[1,0],[2,-2],[2,-1],[2,0]]

            grid_list = [[-2,-2],[-2,-1],[-1,-2],[-1,-1],[0,-2],[0,-1],[1,-2],[1,-1],[2,-2],[2,-1]]

            grid_inc_left = torch.tensor(grid_list).to(device)

            return grid_inc_left # CUDA  torch.Size([14, 2]) dtype: torch.int64

        def increment_full():
            #output-- grid_inc_full: CUDA size (24,2)
            grid_list = [[-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2],[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],[0,-2],[0,-1],[0,1],[0,2],
                         [1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-2],[2,-1],[2,0],[2,1],[2,2]]

            grid_inc_full = torch.tensor(grid_list).to(device)

            return grid_inc_full # CUDA  torch.Size([24, 2]) dtype: torch.int64

        self.grid_inc_1 = increment_1()
        self.grid_inc_2 = increment_2()
        self.grid_inc_3 = increment_3()
        self.grid_inc_4 = increment_4()
        self.grid_inc_up = increment_up()
        self.grid_inc_right = increment_right()
        self.grid_inc_down = increment_down()
        self.grid_inc_left = increment_left()
        self.grid_inc_full = increment_full()



#####################
    def v_to_g(self, index):
        

        grid_x = index // self.grid_x_max
        grid_y = index % self.grid_x_max

        grid_x1 = grid_x.view(1)
        grid_y1 = grid_y.view(1)

        grid_index = torch.cat((grid_x1, grid_y1), dim=0)
        grid_index1 = grid_index.view(1, 2)

        return grid_index1

    def agent_direct_vector(self, direction_vector, grid_index1, exist_agent_last_grid):
        

        grid_deviation = grid_index1 - exist_agent_last_grid  # CUDA  torch.Size([1, 2])  dtype: torch.int64

        deviation1 = torch.where(grid_deviation > 0, self.positive, grid_deviation)

        deviation2 = torch.where(deviation1 < 0, self.negative, deviation1)

        if torch.equal(deviation2, self.quadrant1):
            direction_vector[0, 1] = 1
        elif torch.equal(deviation2, self.quadrant2):
            direction_vector[0,  7] = 1
        elif torch.equal(deviation2, self.quadrant3):
            direction_vector[0, 5] = 1
        elif torch.equal(deviation2, self.quadrant4):
            direction_vector[0, 3] = 1
        elif torch.equal(deviation2, self.quadrant_up):
            direction_vector[0, 0] = 1
        elif torch.equal(deviation2, self.quadrant_right):
            direction_vector[0, 2] = 1
        elif torch.equal(deviation2, self.quadrant_down):
            direction_vector[0, 4] = 1
        elif torch.equal(deviation2, self.quadrant_left):
            direction_vector[0, 6] = 1
        else:
            pass

        return direction_vector  # CUDA   torch.Size([1, 8])   torch.int64

    def agent_direct_control(self, direction_vector):
      

        allow_direction = torch.zeros((1, 8)).long().to(device)
       

        if (direction_vector[0,1] == 1) or (direction_vector[0,0] == 1 and direction_vector[0,2] == 1):
            allow_direction[0, 1] = 1

        elif (direction_vector[0,7] == 1) or (direction_vector[0,0] == 1 and direction_vector[0,6] == 1):
            allow_direction[0, 7] = 1
            

        elif (direction_vector[0, 5] == 1) or (direction_vector[0, 4] == 1 and direction_vector[0, 6] == 1):
            allow_direction[0, 5] = 1
            

        elif (direction_vector[0, 3] == 1) or (direction_vector[0, 2] == 1 and direction_vector[0, 4] == 1):
            allow_direction[0, 3] = 1
            

        elif (direction_vector[0, 0] == 1) and (torch.sum(direction_vector).view(1) == self.only1):
            allow_direction[0, 0] = 1
            

        elif (direction_vector[0, 2] == 1) and (torch.sum(direction_vector).view(1) == self.only1):
            allow_direction[0, 2] = 1
            

        elif (direction_vector[0, 4] == 1) and (torch.sum(direction_vector).view(1) == self.only1):
            allow_direction[0, 4] = 1
           

        elif (direction_vector[0, 6] == 1) and (torch.sum(direction_vector).view(1) == self.only1):
            allow_direction[0, 6] = 1
            
        else:
            pass 
        return allow_direction


    def optional_grids1(self, grid_index1, allow_direction):
        

        if allow_direction[0, 0] == 1:
            grids_allow0 = grid_index1 + self.grid_inc_up
        elif allow_direction[0, 1] == 1:
            grids_allow0 = grid_index1 + self.grid_inc_1
        elif allow_direction[0, 2] == 1:
            grids_allow0 = grid_index1 + self.grid_inc_right
        elif allow_direction[0, 3] == 1:
            grids_allow0 = grid_index1 + self.grid_inc_4
        elif allow_direction[0, 4] == 1:
            grids_allow0 = grid_index1 + self.grid_inc_down
        elif allow_direction[0, 5] == 1:
            grids_allow0 = grid_index1 + self.grid_inc_3
        elif allow_direction[0, 6] == 1:
            grids_allow0 = grid_index1 + self.grid_inc_left
        elif allow_direction[0, 7] == 1:
            grids_allow0 = grid_index1 + self.grid_inc_2
        else:
            
            grids_allow0 = grid_index1 + self.grid_inc_full

        

        grid_index2_max = self.grid_index2_max  

        grids_allow_sign0 = torch.where(grids_allow0 <= grid_index2_max, grids_allow0, self.sign)

        grids_allow_sign = torch.where(grids_allow_sign0 < 0, self.sign, grids_allow_sign0)

        area1 = (grids_allow_sign[:, 0] != -1) & (grids_allow_sign[:, 1] != -1)
        
        grids_allow = grids_allow0[area1] 
        
        return grids_allow

    def g_to_v(self, agent_grids):  # need to change with input as CUDE Tensor
       

        vector_index = agent_grids[:, 0] * self.grid_x_max + agent_grids[:, 1]

        return vector_index

   
    def exi_line_control(self, agent_current_index, vector_index):
        

        if self.exi_sta_adj_sta == None:
            vector_index_allow = vector_index
        else:
            try: 
                grid_exi_mask = self.exi_sta_adj_sta[agent_current_index]  
                #print('grid_exi_mask:',grid_exi_mask)
                #print('grid_exi_mask[0]_type:',type(grid_exi_mask[0]))
            except:
                vector_index_allow = vector_index
            else:
                num = 0
                for i in grid_exi_mask:
                    num = num + 1
                    this_area = (vector_index[:] != i)

                    if num == 1:
                        area = this_area
                    else:
                        area = area & this_area
                vector_index_allow = vector_index[area]  #，vector_index_allow=tensor([], device='cuda:0', dtype=torch.int64)

        return vector_index_allow
    

    


    # add vector_index_allow to 1 vector mask

    def vector_allow(self, agent_current_index, grid_index1, exist_agent_last_grid, direction_vector):
        
        # output--direction_vector:
        #         vector_index_allow:  CUDA,
        #         example1: tensor([2, 3], device='cuda:0')--agent can choose 2 and 3 grids.
        #         example2: tensor([], device='cuda:0')--agent can choose no grids.



        #modified

     #   direction_vector = torch.zeros((1, 8)).long().to(device)

     #   direction_vector = self.agent_direct_vector(direction_vector, grid_index1, exist_agent_last_grid) #current direction

   #     grids_allow = self.optional_grids1(grid_index1, direction_vector)

        direction_vector = self.agent_direct_vector(direction_vector, grid_index1, exist_agent_last_grid)

        allow_direction = self.agent_direct_control(direction_vector)

        grids_allow = self.optional_grids1(grid_index1, allow_direction)

        if grids_allow.size()[0]: 

            vector_index = self.g_to_v(grids_allow)

            vector_index_allow = self.exi_line_control(agent_current_index, vector_index) #

            if not vector_index_allow.size()[0]:  #vector_index_allow: tensor([], device='cuda:0', dtype=torch.int64)
                vector_index_allow = self.null_tensor

        else: # grids_allow =  tensor([], device='cuda:0', dtype=torch.int64) 


            vector_index_allow = self.null_tensor

        return direction_vector, vector_index_allow


    def update_mask(self, vector_index_allow): # focuse  CUDA Tensor
        

        # output-- mask:  CUDA, torch.Size([1, city_number]) torch.float32

        mask_initial = torch.zeros(1, self.grid_num, device=device).long() # 1 : bacth_size


        mask = mask_initial.index_fill_(1, vector_index_allow, 1).float()  # the first 1: dim , the second 1: value

        #mask: example--tensor([[0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1.]],device='cuda:0')
        # size: torch.Size([1, 16])  .dtype: torch.float32
        return mask

    def update_dynamic(self, dynamic, agent_current_index):
       

        h = float(1)


        dynamic = dynamic.clone()
        dynamic[0, 0, agent_current_index] = h

        return dynamic
    

    
    
####################################################################
# define reward

##########
#build od matrix

# od_path =r'/home/weiyu/program/metro_expand_combination/OD.txt'

def local_g_to_v(grid, grid_x_max):
    # grid: axample-- 0,0  string
    # grid_x_max：
    grid_x, grid_y = grid.split(',')

    index = int(grid_x)*grid_x_max + int(grid_y)
    index1 =str(index)

    return index1

def index_od(od_path, grid_x_max, od_index_path):

    f = open(od_path, 'r')
    m = open(od_index_path, 'w')

    for line in f:
        grid1, grid2, weight = line.rstrip().split('\t')

        index1 = local_g_to_v(grid1, grid_x_max)
        index2 = local_g_to_v(grid2, grid_x_max)

        to_write = index1+'\t'+index2+'\t'+weight+'\n'

        m.write(to_write)
    m.close()
    f.close()

#GPU
def build_od_matrix(grid_num, od_index_path):
    
    od_matirx = torch.zeros((grid_num, grid_num)).to(device)

    f = open(od_index_path, 'r')
    for line in f:
        index1, index2, weight = line.rstrip().split('\t')
        index11 = int(index1)
        index21 = int(index2)
        weight1 = float(weight)

        od_matirx[index11][index21] = weight1
    f.close()

    return od_matirx

# od_matirx = build_od_matrix(grid_num, od_index_path)

def agent_pair(agent_grid_list):
   
    # output--satisfied_od_pair:   [[1,2],[2,3]]

    satisfied_od_pair = []

    for i in range(len(agent_grid_list) - 1):
        for j in range(i + 1, len(agent_grid_list)):
            per_od_pair = []
            per_od_pair.append(agent_grid_list[i])
            per_od_pair.append(agent_grid_list[j])
            satisfied_od_pair.append(per_od_pair)

    return satisfied_od_pair

#GPU
def agent_exist_line_pair(tour_idx, agent_grid_list, per_line_full_tensor, per_line_station_list):
   

    satisfied_od_pair = []

    agent_line = (tour_idx - per_line_full_tensor)

    intersection_need = (agent_line == 0).nonzero()

    if intersection_need.size()[0] == 0:
        pass # there is no interaction

    else:
        interaction_index_mult = intersection_need[:, 1]
        interaction_index_list = []
        for i in interaction_index_mult:
            interaction_index_list.append(agent_grid_list[i])

        for i in agent_grid_list:
            if i not in interaction_index_list:
                for j in per_line_station_list:
                    if j not in interaction_index_list:
                        per_od_pair = []
                        per_od_pair.append(i)
                        per_od_pair.append(j)
                        satisfied_od_pair.append(per_od_pair)

    return satisfied_od_pair # for each element: the agent station is the first

# GPU
def satisfied_od_pair_fn(tour_idx, agent_grid_list, line_full_tensor, line_station_list, exist_line_num):
    

    #agent_station_num = len(agent_grid_list)

    satisfied_od_pair1 = agent_pair(agent_grid_list)

    satisfied_od_pair2 = []

    for i in range(exist_line_num):
        per_line_full_tensor = line_full_tensor[i]

        per_line_station_list = line_station_list[i]

        per_satisfied_od_pair2 = agent_exist_line_pair(tour_idx, agent_grid_list, per_line_full_tensor, per_line_station_list)
        

        satisfied_od_pair2 = satisfied_od_pair2 + per_satisfied_od_pair2

    satisfied_od_pair = satisfied_od_pair1 + satisfied_od_pair2

    return satisfied_od_pair  # list cpu

#GPU
def satisfied_od_mask_fn(grid_num, satisfied_od_pair):
    # build the satisfied_od_mask: the element 1(0) present this od is (is not) satisfied by the agent

    satisfied_od_mask = torch.zeros(grid_num, grid_num).byte().to(device)  # initial########################

    value = torch.tensor([1]).byte().to(device)

    for per_pair in satisfied_od_pair:
        i,j = per_pair

        satisfied_od_mask[i][j] = value

    return satisfied_od_mask


# GPU
def reward_fn(tour_idx, grid_num, agent_grid_list, line_full_tensor, line_station_list, exist_line_num, od_matirx):

    satisfied_od_pair = satisfied_od_pair_fn(tour_idx, agent_grid_list, line_full_tensor, line_station_list, exist_line_num)
    # up ok
    satisfied_od_mask = satisfied_od_mask_fn(grid_num, satisfied_od_pair)

    satisfied_od_tensor = torch.masked_select(od_matirx, satisfied_od_mask)

    reward = satisfied_od_tensor.sum()   # CUDA,

    return reward

#########reward cpu
#CPU
def build_od_matrix1(grid_num, od_index_path):
    
    od_matirx = torch.zeros((grid_num, grid_num))

    f = open(od_index_path, 'r')
    for line in f:
        index1, index2, weight = line.rstrip().split('\t')
        index11 = int(index1)
        index21 = int(index2)
        weight1 = float(weight)

        od_matirx[index11][index21] = weight1
    f.close()

    return od_matirx

##CPU
## This part is used to ecxclude od pair

def process_segment(per_segment,grid_x_max):

    per_seg_ind = []
    for i in per_segment:
        grid_x, grid_y = i
        index = grid_x*grid_x_max + grid_y

        per_seg_ind.append(index)
    return per_seg_ind



#CPU
def exlude_od_pair(grid_x_max):

##############################
## consider with only the first and second lines

    line0_nei1 = [[8, 1], [9, 2], [10, 2], [11, 3], [11, 4], [11, 5], [12, 6], [12, 7], [12, 8], [13, 9], [13, 10],
                  [13, 11], [13, 12], [13, 13]]
    line0_nei2 = [[13, 15], [13, 16], [13, 17], [13, 18],[13,19],[13, 20], [12, 21], [12, 22], [12, 23], [12, 24], [12, 25]]
    line0_nei3 = [[8, 3], [9, 4], [9, 5], [10, 6], [10, 7], [10, 8], [11, 9], [11, 10], [11, 11], [11, 12], [11, 13]]
    line0_nei4 = [[11,15], [11,16], [11,17], [11,18],[11, 19], [11, 20], [10, 21], [10, 22], [10, 23], [10, 24], [10, 25]]


#the line1 behine only is without the third and fourth lines
    line1_nei1 = [[0, 12], [1, 13], [2, 13], [3, 13], [4, 13], [5, 13], [6, 13], [7, 13], [8, 13], [9, 13], [10, 13], [11, 13]]
    line1_nei2 = [[13, 13], [14, 13], [15, 13], [16, 13], [17, 13], [18, 13], [19, 13], [20, 13], [21, 13], [22, 13], [23, 13],
                  [24, 13], [25, 13], [26, 13]]

    line1_nei3 = [[0, 14], [1, 15], [2, 15], [3, 15], [4, 15], [5, 15], [6, 15], [7, 15], [8, 15], [9, 15], [10, 15], [11, 15]]

    line1_nei4 = [[13, 15], [14, 15], [15, 15], [16,15], [17, 15], [18, 15], [19, 15], [20, 15], [21, 15], [22, 15], [23, 15],
                  [24, 15], [25, 15], [26, 15]]
    

    segment_list = []
    segment_list.append(line0_nei1)
    segment_list.append(line0_nei2)
    segment_list.append(line0_nei3)
    segment_list.append(line0_nei4)


    segment_list.append(line1_nei1)
    segment_list.append(line1_nei2)
    segment_list.append(line1_nei3)
    segment_list.append(line1_nei4)

    #new line
    
    line2_nei = [[3, 26], [5, 24], [6, 22], [6, 21], [7, 21], [7, 20], [7, 19], [8, 19], [9, 19], [9, 18], [10, 18], [11, 18], [11, 17], [12, 17], [13, 17], [13, 16], 
                           [13,15], [13, 14], [13, 13], [14, 13], [15, 13], [15, 12], [16, 12], [16, 11], [17, 11], [17, 10], [17, 9], [18, 9], [19, 9], [19, 8], [19, 7], [20, 7], [21, 7], [22, 7], [22, 6]]
    
    line3_nei =  [[5, 0], [7, 0], [8, 1], [8, 2], [9, 2], [9, 3], [9, 4], [10, 4], [11, 4], [11, 5], [11, 6], [12, 6], [12, 7], [13, 7], [13, 8], [14, 8], [14, 9], 
                               [14, 10], [15, 10], [15, 11], [16, 11], [17, 12], [17, 13], [17, 14], [17, 15], [18, 15], [18, 16], [19, 16], [19, 17], [20, 17], [20, 18], [20, 19], [21, 19], [21, 20], [22, 20]]
    
    segment_list.append(line2_nei)
    segment_list.append(line3_nei)
    


    segment_vec_index = []
    for per_segment in segment_list:
        per_seg_ind = process_segment(per_segment,grid_x_max)
        segment_vec_index.append(per_seg_ind)

    exclude_pair = []
    for per_seg_ind in segment_vec_index:

        for i in range(len(per_seg_ind)-1):
            for j in range(i+1, len(per_seg_ind)):
                per_pair = [per_seg_ind[i], per_seg_ind[j]]
                per_pair1 = [per_seg_ind[j], per_seg_ind[i]]

                exclude_pair.append(per_pair)
                exclude_pair.append(per_pair1)
    # exclude_pair1 = list(set(exclude_pair))

    return exclude_pair


def od_matrix_exclude(od_matirx, exclude_pair):

    for per_pair in exclude_pair:
        i, j = per_pair

        od_matirx[i][j] = 0.0

    return od_matirx



#CPU
def agent_exist_line_pair1(tour_idx_cpu, agent_grid_list, per_line_full_tensor, per_line_station_list):
   

    satisfied_od_pair = []

    agent_line = (tour_idx_cpu - per_line_full_tensor)

    intersection_need = (agent_line == 0).nonzero()

    if intersection_need.size()[0] == 0:
        pass # there is no interaction

    else:
        interaction_index_mult = intersection_need[:, 1]
        interaction_index_list = []
        for i in interaction_index_mult:
            interaction_index_list.append(agent_grid_list[i])

        for i in agent_grid_list:
            if i not in interaction_index_list:
                for j in per_line_station_list:
                    if j not in interaction_index_list:
                        per_od_pair = []
                        per_od_pair.append(i)
                        per_od_pair.append(j)
                        satisfied_od_pair.append(per_od_pair)

    return satisfied_od_pair # for each element: the agent station is the first

#CPU
def min_dis_od(satisfied_od_pair, grid_x_max, dis_lim):
   
    # output--true_satisfied_od_pair: 

    true_satisfied_od_pair = []

    if satisfied_od_pair: # there are interaction stations

        satisfied_od_pair_tensor = torch.tensor(satisfied_od_pair)

        grid_x_tensor = satisfied_od_pair_tensor // grid_x_max
        grid_y_tensor = satisfied_od_pair_tensor % grid_x_max

        dis_x = grid_x_tensor[:, 1] - grid_x_tensor[:, 0]
        dis_y = grid_y_tensor[:, 1] - grid_y_tensor[:, 0]

        dis_tensor = (dis_x.pow(2) + dis_y.pow(2)).float().sqrt()
        od_index = (dis_tensor > dis_lim).nonzero()

        if od_index.size()[0] == 0:
            pass  # there is no satisfied_od pair
        else:
            satisfied_od_index = od_index[:, 0]

            for i in satisfied_od_index:
                true_satisfied_od_pair.append(satisfied_od_pair[i])

    return true_satisfied_od_pair



# CPU
def satisfied_od_pair_fn1(tour_idx_cpu, agent_grid_list, line_full_tensor, line_station_list, exist_line_num, grid_x_max, dis_lim):
    

    satisfied_od_pair1 = agent_pair(agent_grid_list)

    if dis_lim == -1: #od pairs in reward only consider agent line
        satisfied_od_pair = satisfied_od_pair1

    else:

        satisfied_od_pair2 = []

        for i in range(exist_line_num):
            per_line_full_tensor = line_full_tensor[i]

            per_line_station_list = line_station_list[i]

            per_satisfied_od_pair2 = agent_exist_line_pair1(tour_idx_cpu, agent_grid_list, per_line_full_tensor, per_line_station_list)
            

            if dis_lim:
               
                per_true_satisfied_od_pair = min_dis_od(per_satisfied_od_pair2, grid_x_max, dis_lim)

                satisfied_od_pair2 = satisfied_od_pair2 + per_true_satisfied_od_pair

            else: 
                satisfied_od_pair2 = satisfied_od_pair2 + per_satisfied_od_pair2

        satisfied_od_pair = satisfied_od_pair1 + satisfied_od_pair2

    return satisfied_od_pair  # list cpu

#CPU
def satisfied_od_mask_fn1(grid_num, satisfied_od_pair):
    # build the satisfied_od_mask: the element 1(0) present this od is (is not) satisfied by the agent

    satisfied_od_mask = torch.zeros(grid_num, grid_num).byte()  # initial########################

    value = torch.tensor([1]).byte()

    for per_pair in satisfied_od_pair:
        i, j = per_pair

        satisfied_od_mask[i][j] = value

    return satisfied_od_mask

#CPU
def reward_fn1(tour_idx_cpu, grid_num, agent_grid_list, line_full_tensor, line_station_list, exist_line_num, od_matirx, grid_x_max, dis_lim):

    satisfied_od_pair = satisfied_od_pair_fn1(tour_idx_cpu, agent_grid_list, line_full_tensor, line_station_list, exist_line_num, grid_x_max, dis_lim)
    # up ok
    satisfied_od_mask = satisfied_od_mask_fn1(grid_num, satisfied_od_pair)
    satisfied_od_mask = satisfied_od_mask.ge(0.5)

    satisfied_od_tensor = torch.masked_select(od_matirx, satisfied_od_mask)

    reward = satisfied_od_tensor.sum()   # CPU

    return reward





#CPU

def build_grid_price(path_house, grid_x_max, grid_y_max):
#input--path_house: r'/home/weiyu/program/metro_expand_combination/index_average_price.txt'
    # price_matrix = torch.zeros((grid_y_max, grid_x_max)).to(device)
    price_matrix = torch.zeros((grid_y_max, grid_x_max)).float()

    f = open(path_house, 'r')

    for line in f:
        grid,price = line.rstrip().split('\t')
        index_x,index_y = grid.split(',')

        index_x = int(index_x)
        index_y = int(index_y)

        price_matrix[index_x][index_y] = float(price)
    f.close()
    return price_matrix

# # Utilitarianism
def agent_grids_price(tour_idx_cpu, grid_x_max, price_matrix):
    agent_grids_num = tour_idx_cpu.size()[1]

    grid_x = tour_idx_cpu // grid_x_max
    grid_y = tour_idx_cpu % grid_x_max
    grid_index = torch.cat((grid_x, grid_y), dim=0)

    grids = grid_index.transpose(0, 1)  # torch.int64

    tour_idx_price = torch.zeros((agent_grids_num, 1)).float()

    for i in range(agent_grids_num):
        per_grid = grid_index[:, i]
        gridx = per_grid[0]
        gridy = per_grid[1]

        per_price = price_matrix[gridx, gridy]
        tour_idx_price[i][0] = per_price

    Ac = []

    for i in range(agent_grids_num):
        per_grid_expand = grids[i].expand_as(grids)

        per_need = per_grid_expand - grids

        per_need1 = per_need.pow(2)

        per_need2 = per_need1.sum(dim=1).float()

        per_need3 = per_need2.sqrt()

        per_need4 = torch.exp(-0.5 * per_need3)


        per_need5 = per_need4 * (tour_idx_price.transpose(0, 1))

        per_Ac = per_need5.sum()

        Ac.append(per_Ac)
    agent_Ac = sum(Ac)

    return agent_Ac

# #Equal Sharing
def agent_grids_price1(tour_idx_cpu, grid_x_max, price_matrix):
    agent_grids_num = tour_idx_cpu.size()[1]

    grid_x = tour_idx_cpu // grid_x_max
    grid_y = tour_idx_cpu % grid_x_max
    grid_index = torch.cat((grid_x, grid_y), dim=0)

    grids = grid_index.transpose(0, 1)  # torch.int64

    tour_idx_price = torch.zeros((agent_grids_num, 1)).float()

    for i in range(agent_grids_num):
        per_grid = grid_index[:, i]
        gridx = per_grid[0]
        gridy = per_grid[1]

        per_price = price_matrix[gridx, gridy]
        tour_idx_price[i][0] = per_price

    Ac = []

    for i in range(agent_grids_num):
        per_grid_expand = grids[i].expand_as(grids)

        per_need = per_grid_expand - grids

        per_need1 = per_need.pow(2)

        per_need2 = per_need1.sum(dim=1).float()

        per_need3 = per_need2.sqrt()

        per_need4 = torch.exp(-0.5 * per_need3)

        per_need4[i] = 0  # the increase needs exclude the owner.

        per_need5 = per_need4 * (tour_idx_price.transpose(0, 1))

        per_Ac = per_need5.sum()
        per_Ac1 = per_Ac.view(1)

        Ac.append(per_Ac1) # Ac example: [tensor(0.), tensor(0.), tensor(0.), tensor(0.), tensor(0.)]

    average_Ac = sum(Ac) / agent_grids_num

    Ac_tensor = torch.cat(Ac, dim=0)

    total_diff_sum = torch.tensor(0.0)
    for i in range(agent_grids_num):
        per_difference = Ac_tensor[i].view(1)-Ac_tensor
        per_diff_abs = torch.abs(per_difference, out = None)
        per_diff_sum = per_diff_abs.sum()
        total_diff_sum = total_diff_sum + per_diff_sum
    try:
        pi = math.pi
        agent_Ac = total_diff_sum / (2*pi*pi*average_Ac)

        # agent_Ac = total_diff_sum / (2 * agent_grids_num * agent_grids_num * average_Ac)
        agent_Ac = agent_Ac.data[0]
    except: #average_Ac may be 0
        agent_Ac = torch.tensor(0.0)
    finally:
        if torch.isnan(agent_Ac):  
            agent_Ac = torch.tensor(0.0)
    return agent_Ac



####################################################################
# calculate radiation

def find_intersection(new, ExisLineStri):

      new = new.cpu().detach().numpy()

      new_line = LineString(np.column_stack((new[0], new[1])))

      a = shapely.intersection(new_line, ExisLineStri[0])

      intersection1 = np.array(shapely.intersection(new_line, ExisLineStri[0]).bounds)

   #   intersection1 = np.array((new_line.intersection(ExisLineStri[0])).coords)
      intersection2 = np.array((shapely.intersection(new_line, ExisLineStri[1])).bounds)


      intersection1_re = np.array([intersection1[0], intersection1[1]])
      intersection2_re = np.array([intersection2[0], intersection2[1]])

      intersection = [intersection1_re, intersection2_re]


      return intersection

def BuildDict (InterList, graph, new_line, inter_g, ExisLineStri, ExisIndexList, line_full_tensor, grid_x_max):

    NewInterIndex, NewInter= [], []

    new_line = new_line.cpu().detach().numpy().tolist()[0]

    graph[new_line[0]], graph[new_line[-1]] = [], []

    


    for i in range(len(InterList)):

        CheckList = [inter_g, line_full_tensor[i][0], line_full_tensor[i][-1]]

        if InterList[i][0] and not math.isnan(InterList[i][0]):


                new_inter_ = int(InterList[i][0] * grid_x_max + InterList[i][1])
            
    ###the location of the new intersection
                if new_inter_ in new_line and new_inter_ in line_full_tensor[i]:

                    new_inter = new_inter_

                    if new_inter not in CheckList and new_inter not in NewInter:

                        graph[new_inter] = []
                    
                    
                    new_inter_index = line_full_tensor[i].index(new_inter)



                    NewInterIndex.append(new_inter_index), NewInter.append(new_inter)

                    graph[inter_g].append(new_inter),graph[new_inter].append(inter_g)


                    if new_inter in ExisIndexList[i] :
                        

                        graph[new_inter].append(line_full_tensor[i][0])
                        graph[line_full_tensor[i][0]].append(new_inter)
                        if new_inter not in CheckList and inter_g in graph[line_full_tensor[i][0]] and line_full_tensor[i][0] in graph[inter_g]:
                            graph[line_full_tensor[i][0]].remove(inter_g)
                            graph[inter_g].remove(line_full_tensor[i][0])
                    
                    else:
                        graph[new_inter].append(line_full_tensor[i][-1])
                        graph[line_full_tensor[i][-1]].append(new_inter)
                        if new_inter not in CheckList and inter_g in graph[line_full_tensor[i][-1]]:
                            graph[line_full_tensor[i][-1]].remove(inter_g)


                else:
                    NewInterIndex.append(None), NewInter.append(None)

        else:
            NewInterIndex.append(None), NewInter.append(None)

    if not all(i is None for i in NewInter):

            if any(j is None for j in NewInter) or NewInter[0] == NewInter[1]:

                    graph[new_inter].append(new_line[0])
                    graph[new_inter].append(new_line[-1])
                    graph[new_line[0]].append(new_inter), graph[new_line[-1]].append(new_inter)
                    graph[new_inter].append(inter_g)

            else:

                graph[NewInter[0]].append(NewInter[1])
                graph[NewInter[1]].append(NewInter[0])
                
                inter_index = new_line.index(NewInter[0])
                new_slice = new_line[:inter_index]

                if NewInter[i] in new_slice:
                    graph[new_line[0]].append(NewInter[1]), graph[NewInter[1]].append(new_line[0])
                    graph[new_line[-1]].append(NewInter[0]), graph[NewInter[0]].append(new_line[-1])
                else:
                    graph[new_line[0]].append(NewInter[0]), graph[NewInter[0]].append(new_line[0])
                    graph[new_line[-1]].append(NewInter[1]), graph[NewInter[1]].append(new_line[-1])

    else:
        graph[new_line[0]].append(new_line[-1]), graph[new_line[-1]].append(new_line[0])

   # print(NewInter, graph, '\n', line_full_tensor,'\n', new_line )
   # print(graph)
    for key in graph.keys():
        graph[key] = list(set(graph[key] ))
    return graph

def findAllPath(graph,start,end, paths, path=[]):
    #path.append(start)
    path = path + [start]
    if start == end:
        paths.append(path)
        return paths
 
    for node in graph[start]:
        if node not in path:
            newpaths = findAllPath(graph,node,end,paths, path) 
            for newpath in newpaths:
                if newpath==end:
                    path.append(newpath)
                    paths.append(path)
                    
    return paths


def radiation_ac(new_line, grid_num, agent_grid_list, line_full_tensor, line_station_list, exist_line_num, od_matirx, grid_x_max, dis_lim, ExisIndexList, inter_g, ExisLineStri):

    graph = dict()

    graph[line_station_list[0][0]], graph[line_station_list[0][-1]], graph[line_station_list[1][0]], graph[line_station_list[1][-1]] = \
    [inter_g], [inter_g], [inter_g], [inter_g]



    graph[inter_g] = [line_station_list[0][0]]
    graph[inter_g].append(line_station_list[0][-1])
    graph[inter_g].append(line_station_list[1][0])
    graph[inter_g].append(line_station_list[1][-1])



    grid_x, grid_y = new_line // grid_x_max, new_line % grid_x_max

    reward_radia = 0

    station_num = grid_x.size()[1]

    if station_num > 2 :

        gird_new_line = torch.cat((grid_x, grid_y), dim=0)

        InterList = find_intersection(gird_new_line, ExisLineStri)


        graph = BuildDict(InterList, graph, new_line, inter_g, ExisLineStri, ExisIndexList, line_station_list, grid_x_max)

        station_num = len(graph)

        for key in graph.keys():
            start = key

            station_radia = 0

            for j in graph.keys():

                end = j

                paths, path_length = [], 20

                paths = findAllPath(graph, start, end, paths)

    #          print(start, end, paths)

                if len(paths):

                    path_length = min([len(i) for i in paths])

                station_radia += path_length

            AD_i = 1/(station_num-1) * station_radia

            AI_i = (station_num-2)/ (2* (AD_i -1))

            reward_radia += AI_i



    return torch.tensor(reward_radia * 3.5)


#build static
        #RLSM dont have the weight information
def build_static(grid_x_max, grid_y_max, weight, method, grid_num, static_size):
            
          
    for i in range(grid_y_max):
            per_need = np.zeros((grid_x_max, 2))
            
            for j in range(grid_x_max):
                per_need[j, 0] = i
                per_need[j, 1] = j

            if i == 0:
                need = per_need
            else:
                need = np.vstack((need, per_need))

    np_static = need.transpose()
    weight = weight.transpose()
        
    if method != 'RLSM':
        weight_need = np.full([3,grid_x_max * grid_y_max], weight)
        np_static = np.vstack((np_static,weight_need))
    

    ten_static = torch.from_numpy(np_static).float()
    static = ten_static.view(1, static_size, grid_num).to(device) # size with batch

    return static



def radiation_ac_a(new_line, grid_num, agent_grid_list, line_full_tensor, line_station_list, exist_line_num, od_matirx, grid_x_max, dis_lim, ExisIndexList, inter_g, ExisLineStri):
    
    graph = dict()

    new_line = new_line.squeeze().numpy().tolist()

    graph[line_station_list[0][0]], graph[line_station_list[0][-1]], graph[line_station_list[1][0]], graph[line_station_list[1][-1]] = \
    [inter_g], [inter_g], [inter_g], [inter_g]



    graph[inter_g] = [line_station_list[0][0]]
    graph[inter_g].append(line_station_list[0][-1])
    graph[inter_g].append(line_station_list[1][0])
    graph[inter_g].append(line_station_list[1][-1])

    initial_data = syntax(graph)

    #储存每条线上的拓扑点

    syntax0 = sorted([line_station_list[0][0],line_station_list[0][-1],inter_g])

    syntax1 = sorted([line_station_list[1][0],line_station_list[1][-1],inter_g])

    syntax_list = [syntax0,syntax1]




    #查找新线与既有线的交





    ret0 = [i for i in new_line if i in line_station_list[0]]
    ret1 = [i for i in new_line if i in line_station_list[1]]

    new_syntax =[new_line[0],new_line[-1]]



    ret = [ret0,ret1]

    if len(ret0) or len(ret1):
        graph, new_syntax = update(ret,graph, syntax_list, line_station_list[0],line_station_list[1],new_syntax) 

    #更新拓扑图

      



    return torch.tensor(2 * 3.5)


def update(ret, network,syntax, line0,line1,new):
 
    for i in range(len(ret)):
        syntax_sub = syntax[i]

        for j in range(len(syntax_sub)-1):
            
            for m in range(len(ret[i])):
            
              if ret[i][m] in range(syntax_sub[j],syntax_sub[j+1]):
                  network[ret[i][m]] = [syntax_sub[j],syntax_sub[j+1]]
                  new.append(ret[i][m])

    sorted(new)

    #return 

    return network, new

              

            
            

def syntax(network):
    
    input_data =[]
    
    for key in network:

        for i in range(len(network[key])):

          key_data =[]

          key_data.append(key)
                     
          key_data.append(network[key][i])

          key_data.append(1)

          input_data.append(key_data)

    m = len(input_data)
    n = len(network)

    input_data.insert(0,[n,m])

    
    
    return input_data


        
    

















