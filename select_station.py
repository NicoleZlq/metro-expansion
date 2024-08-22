
import pandas as pd
import numpy as np

import numpy as np
import torch
import cv2
import os

import seaborn as sns
from PIL import Image 


import matplotlib.pyplot as plt


from matplotlib.pyplot import MultipleLocator


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = os.getcwd()

od_index_path = path + '/od_index_dyna.txt'
network_path= path + '/pic/network.png'
price_path = path + '/index_average_price_dyna.txt'
writer = pd.ExcelWriter(path + '/valid_station.xlsx')

grid_x_max, grid_y_max = 29, 29
grid_num = 29 * 29




#透明度
def transPNG(srcImageName):
    img = Image.open(srcImageName)
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = list()
    for item in datas:
        if item[0] > 220 and item[1] > 220 and item[2] > 220:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img

#blend
def mix(img1,img2):
    # im = img1
    # mark = img2
    # layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
    # layer.paste(mark, coordinator)
    # #out = Image.composite(layer, im, layer)
    # out = Image.alpha_composite(im, mark)
    # return out
    img1 = img1.convert('RGBA')
    img2 = img2.convert('RGBA')
    final = Image.new("RGBA", img1.size)             # 合成的image
    final = Image.alpha_composite(final, img1)
    final = Image.alpha_composite(final, img2)

    final=final.convert('RGB')
    return final


def generate_od_matrix(grid_num, od_index_path):
    f = open(od_index_path, 'r')
    od_num = []
     
    od_matrix = np.zeros((grid_num, grid_num))
    for line in f:
        index1, index2, weight = line.rstrip().split('\t')
        index11 = int(index1)
        index21 = int(index2)
        weight1 = float(weight)
        od_num.append(int(weight))

        od_matrix[index11][index21] = weight1
    f.close()


    od_num = list(set(od_num))

    od_per = np.percentile(od_num, 0)

    where_res=np.where(od_matrix>od_per)

    od_station = list(set(list(where_res[0]) + list(where_res[1])))

    index_x1 = []
    index_y1 = []

    for index in od_station:

            grid_x = index // grid_x_max
            grid_y = index %  grid_x_max

            index_x1.append(grid_x)
            index_y1.append(grid_y)


    index_x_v = []
    index_y_v = []


    for i in range(2, len(index_x1),2):
        vec1 = np.stack(np.array([index_x1[i],index_y1[i]]))
        vec2 = np.stack(np.array([index_x1[i-3],index_y1[i-3]]))
        dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
        if dist > 1.5:
                index_x_v.append(index_y1[i-1])
                index_y_v.append(index_x1[i-1])
    print(len(index_x_v), len(index_x1))

    index_x_v, index_y_v = np.array(index_x_v), np.array(index_y_v)

    index_x1, index_y1 = np.array(index_x1), np.array(index_y1)


    return index_x1, index_y1, od_matrix #index_x_v, index_y_v


def plot_heatmap(index_x, index_y, od_matrix, price):
    od_demand = np.zeros((29,29))
    od_list = []
    
        
    valid_station_grid = np.vstack((np.array(index_x),np.array(index_y)))

    valid_station_index = valid_station_grid[0] *29 + valid_station_grid[1]

    od_sum_1 = np.sum(od_matrix, axis=0)
    od_sum_2 = np.sum(od_matrix, axis=1)

    for i in valid_station_index:
       od_list.append(od_sum_1[i] + od_sum_2[i])

    for i in range(len(index_x)):
        od_demand[index_x[i], index_y[i]] = od_list[i]


    ax = sns.heatmap(od_demand,fmt="d", cmap= 'GnBu')

    print(1111)
    figure = ax.get_figure()
    heat_path = 'E:/UMstudy/Paper manu/part C/Metro-Line/pic/OD_heatmap_dyna.png'
    figure.savefig(heat_path) 
    plt.close()


    ax = sns.heatmap(price_matrix,fmt="d", cmap= 'Oranges')


    figure = ax.get_figure()
    heat_path = 'E:\\home\\liqing\\MORL\\Metro-Line\\pic\\price_dyna_heatmap.png'
    figure.savefig(heat_path) 
    plt.close()

    img = Image.open('/home/liqing/MORL/Metro-Line/Metro-Line/pic/network_gray.png')
    img = img.convert('RGBA')
    img_blender = Image.new('RGBA', img.size, (0,0,0,0))
    img = Image.blend(img_blender, img, 0.7)
    img = np.asarray(img)
    cv2.imwrite('/home/liqing/MORL/Metro-Line/Metro-Line/pic/network_gray_trans.png', img)








def initial_final_station(corrider):

    corridor_station_index = []


    if corrider == 1:
        print(1111)
        corridor_station = [29, 0]
        for j in range(29):
            for i in range(4):
                if corridor_station[0] < 29 and corridor_station[0] >=0:
                  corridor_station_c = corridor_station.copy()
                  corridor_station_index.append(corridor_station_c[0]*grid_x_max + corridor_station_c[1])
                corridor_station[0]-=1
            corridor_station[0] +=3
            corridor_station[1] +=1

              
    
    elif corrider == 2:
        print(222)
        corridor_station = [30,0]
        for j in range(29):
            for i in range(6):
                if corridor_station[0] < 29 and corridor_station[0] >=0:
                  corridor_station_c = corridor_station.copy()
                  corridor_station_index.append(corridor_station_c[0]*grid_x_max + corridor_station_c[1])
                corridor_station[0]-=1
            corridor_station[0] +=5
            corridor_station[1] +=1
        print(222)

    elif corrider == 3:
        print(333)
        corridor_station = [-2,0]
        for j in range(29):
            for i in range(6):
                if corridor_station[0] < 29 and corridor_station[0] >=0:
                  corridor_station_c = corridor_station.copy()

                  corridor_station_index.append(corridor_station_c[0]*grid_x_max + corridor_station_c[1])
                corridor_station[0]+=1
            corridor_station[0] -=5
            corridor_station[1] +=1
        
       
    return corridor_station_index

def plot_resize(img1,img2):
     
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)
    image = cv2.resize(image1, (800, 800), interpolation=cv2.INTER_CUBIC)
    image_path_1 = '/home/liqing/MORL/Metro-Line/Metro-Line/pic/scatter_new.png'
    cv2.imwrite(image_path_1, image)
    image = cv2.resize(image2, (800, 800), interpolation=cv2.INTER_CUBIC)
    image_path_2 = '/home/liqing/MORL/Metro-Line/Metro-Line/pic/network_new.png'
    cv2.imwrite(image_path_2, image)

def plot_grey(img):
     
     img = cv2.imread(img)
     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

     out=255-gray
     image_path_3 = '/home/liqing/MORL/Metro-Line/Metro-Line/pic/network_gray.png'

     cv2.imwrite(image_path_3,out)

     return image_path_3

def plot_blend(verse, img1, img2):
     
    file1 = Image.open(img1)
    file2 = Image.open(img2)


    file1 = mix(file1, verse)
    file2 = mix(file2, verse)

    file1.save('/home/liqing/MORL/Metro-Line/Metro-Line/pic/corridor{}.png'.format(str(corridor)))

    #file1.save('/home/liqing/MORL/Metro-Line/Metro-Line/pic/initial{}_pareto{}_budget{}.png'.format(str(corridor),str(args_pareto),str(args_budget)))
    file2.save('/home/liqing/MORL/Metro-Line/Metro-Line/pic/lines_b_corridor{}.png'.format(str(corridor)))

def existing_lines():

    line0_ststion_list = np.array([[8, 2], [10, 3], [10, 5], [11, 6], [11, 7], [12, 9], [12, 11], [12, 12], [12, 13],
                              [12, 14], [12, 15], [12, 16], [12, 17], [12, 18], [12, 20], [11, 22], [11, 23], [11, 24],
                              [11, 25]]).T

    line1_ststion_list = np.array([[0, 13], [1, 14], [3, 14], [4, 14], [5, 14], [6, 14], [8, 14], [9, 14], [10, 14],
                              [12, 14], [13, 14], [14, 14], [15, 14], [16, 14], [17, 14], [18, 14], [20, 14], [21, 14],
                              [23, 14], [24, 14], [26, 14]]).T

    line2_ststion_list = np.array([[15, 5], [15, 7], [15, 9], [16, 10], [17, 11], [17, 13], [17, 14], [17, 16], [17, 17],
                               [16, 18], [15, 19], [14, 19], [13, 19], [11, 18], [9, 18], [8, 18], [7, 20], [7, 22], [7, 23],
                               [5, 24], [3, 23], [2, 22], [0, 23]]).T
        
    line3_ststion_list = np.array([[0, 13], [1, 12], [2, 12], [3, 12], [4, 13], [4, 14], [4, 15], [4, 16], [5, 16], [6, 16],
                               [7, 16], [9, 16], [10, 16], [11, 16], [12, 15], [13, 15], [14, 15], [15, 16], [16, 16], [17, 16],
                               [18, 16], [20, 17], [22, 17], [23, 17], [24, 17], [25, 17], [26, 17], [26, 19], [26, 20]]).T

    line0_ststion_list[0], line1_ststion_list[0] = np.abs(line0_ststion_list[0]-28),np.abs(line1_ststion_list[0]-28)

    line2_ststion_list[0], line3_ststion_list[0] = np.abs(line2_ststion_list[0]-28),np.abs(line3_ststion_list[0]-28)

    

    x_list = [line0_ststion_list[0], line1_ststion_list[0], line2_ststion_list[0], line3_ststion_list[0]]

    y_list = [line0_ststion_list[1], line1_ststion_list[1], line2_ststion_list[1], line3_ststion_list[1]]

    

    return x_list, y_list

def plot_scatter(index_x, index_y, existing_x, existing_y, initial, final):
    color = ['#ef476f', '#ffd166', '#118AD5', '#FFADAD', '#FF6666', '#FBA46A', '#FFD6ED', '#FDFFB6', '#CAFFBF',
         '#9BF6FF', '#A0C4FF', '#BDB2FF', '#FFC6FF', '#C5D188', '#81E3AC', '#A0EBD8', '#74DEE8', '#8FBAFF', 
         '#9D8CFF', '#FFB3FF', '#FA7C5C', '#B9C295','#82D177']
    
    c = [color[i] for i in range(10)]


    label_list= ["candidate station", "existing station-1", "existing station-2" ]  # 生成A标签


     
    plt.figure(figsize=(25, 25), dpi=53)
    plt.margins(x=0)
    plt.margins(y=0)
    #plt.xlabel("x - label")
    #plt.ylabel("y - label")

    plt.xticks(np.linspace(0,29,29,endpoint=True))
    plt.yticks(np.linspace(0,29,30,endpoint=True))

    plt.scatter(np.abs(index_x-28), index_y, c='blue', label = label_list[0], s =400)  # 绘制散点图

    plt.plot(existing_y[0], existing_x[0], 'bo-',color = c[5], label='m=10, p=10%',linewidth=6, markersize='30')
    plt.plot(existing_y[1], existing_x[1], 'bo-', color = c[5],  linewidth=6, label=label_list[2],markersize='30')
    #plt.plot(existing_y[2], existing_x[2], 'bo-', color = c[2],  linewidth=6, label=label_list[2],markersize='30')
    #plt.plot(existing_y[3], existing_x[3], 'bo-', color = c[2],  linewidth=6, label=label_list[2],markersize='30')

    plt.scatter(initial[1], np.abs(initial[0]-28),  c='red', marker='H', label = label_list[0], s =800)  # 绘制散点图
    plt.scatter( final[1],  np.abs(final[0]-28),c='red', marker='H', label = label_list[0], s =800)


    corridor1_x, corridor1_y, corridor2_x, corridor2_y = [], [], [], []

    if corridor == 1:
        x_1, y_1, x_2, y_2 = 0, 2, 1, 0
    elif corridor == 2:
        x_1, y_1, x_2, y_2 = 0, 3, 2, 0
    elif corridor == 3:
        x_1, y_1, x_2, y_2 = 0, 25, 2, 28
    
    for i in range(29):
    
        if y_1 < 29 and y_1 >=0 :
            corridor1_x.append(x_1), corridor1_y.append(y_1)
        if x_2 < 29:
            corridor2_x.append(x_2), corridor2_y.append(y_2)
        
        x_1 +=1
        x_2 +=1

        if corridor == 1 or corridor == 2:
            
            y_1 +=1
            y_2 +=1

        elif corridor == 3:

            y_1 -=1
            y_2 -=1



    
    plt.plot(corridor1_x, corridor1_y, 'bo-', color = c[0],  linewidth=6, label=label_list[2],markersize='0')
    plt.plot(corridor2_x, corridor2_y, 'bo-', color = c[0],  linewidth=6, label=label_list[2],markersize='0')

    plt.margins(x=0)
    plt.margins(y=0)

    
            




    plt.axis('off')
    plt.grid(axis='x',linestyle='-.',linewidth=1,color='black',alpha=0.5)
    plt.grid(axis='y',linestyle='-.',linewidth=1,color='black',alpha=0.5)

    scatter_path = '/home/liqing/MORL/Metro-Line/Metro-Line/pic/scatter_{}.png'.format(str(corridor))



    plt.savefig(scatter_path, bbox_inches='tight', pad_inches=0.02, dpi=800)

    plt.close()

    

    #plt.close()

    return scatter_path


   



def build_grid_price(path_house):
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


def plot_new_station(initial, new_station_path, pareto, budget, scatter_path):
    corridor_path = '/home/liqing/MORL/Metro-Line/Metro-Line/pic/lines_a_corridor{}.png'.format(str(initial))

    

    f = open(new_station_path, 'r')

    index_x, index_y = [], []

    for line in f: 

        result = list(line.split(','))
    
    for index in result:

        index = int(index)

        grid_x = index // 29
        grid_y = index %  29

        index_x.append(grid_x)
        index_y.append(grid_y)

    index_x, index_y = np.array(index_x), np.array(index_y)



    f.close()


    plt.plot(index_y, np.abs(index_x-28), 'bo-',color = 'red',linewidth=6, markersize='30')


  #  plt.scatter(np.abs(index_x-28), index_y, c='green', s =400)  # 绘制散点图

    plt.axis('off')


    plt.savefig(scatter_path,bbox_inches='tight', pad_inches=0.02) 

    plt.close()
    

 #   plt.savefig(scatter_path,bbox_inches='tight', pad_inches=0.02) 

    

    # file1 = Image.open(scatter_path)
    # file2 = Image.open(corridor_path)


    # file = mix(file1, file2)

    # file.save('/home/liqing/MORL/Metro-Line/Metro-Line/result/new_station/initial{}_pareto{}_budget{}.png'.format(str(corridor), str(pareto),str(budget)))

def plot_training(result_path):

    training_path = os.path.join(result_path,'reward_actloss_criloss.txt')

    f = open(training_path, 'r')

    od, equity = [], []


    for line in f: 

        result = list(line.split('\t'))

        od.append(float(result[3]))
        equity.append(float(result[4]))

    od, equity = np.array(od), np.array(equity)
    
    # plt.subplot(2, 1, 1)
    plt.plot(od, '-', label="Satisfied OD demand")
    plt.plot(equity, '-', label="Social equity")
    #plt.title('pareto_{}_initial{}_budget{}'.format(str(args_pareto),str(corridor),str(args_budget)))
    plt.ylabel('Reward')
    plt.xlabel('Epoch')
    plt.legend(loc='center')
    plt.margins(x=0)
    plt.margins(y=0)

    y_major_locator=MultipleLocator(4)
    
    ax=plt.gca()

    
    ax.yaxis.set_major_locator(y_major_locator)


    # plt.subplot(2, 1, 2)
    # plt.plot(critic_loss_list, 'o-', label="critic_loss")
    # plt.xlabel('Critic_loss vs. epoches')
    # plt.ylabel('Critic loss')
    # plt.legend(loc='best')
    picture_path = os.path.join(result_path, 'loss_pareto_{}_initial{}_budget{}'.format(str(args_pareto),str(corridor),str(args_budget)))

    plt.savefig(picture_path, dpi=800)

    plt.close()



index_x, index_y, od_matrix = generate_od_matrix(grid_num, od_index_path)


price_matrix = build_grid_price(price_path)

plot_heatmap(index_x, index_y, od_matrix, price_matrix)




     
# types of corrider + initial station
# corridor, args_pareto, args_budget = 1, 'True', 270 

# result_path = '/home/liqing/MORL/Metro-Line/Metro-Line/result/new_station/tour_idx_initial1_paretoTrue_budget270'

# training_path ='/home/liqing/MORL/Metro-Line/Metro-Line/result/20_40_04.306632/pareto_True_initial_1_budget_270_w0.5'



# existing_x, existing_y =  existing_lines()

# corridor_index = initial_final_station(corridor)


# scatter_path = plot_scatter(index_x, index_y, existing_x, existing_y) 


# #plot_new_station(corridor, result_path, args_pareto, args_budget, scatter_path)


# #plt.savefig(scatter_path,bbox_inches='tight', pad_inches=0.02) 

# # plt.close()


# plot_resize(scatter_path,network_path)

# #network_new_path ='/home/liqing/MORL/Metro-Line/Metro-Line/pic/network_new.png'

# #network_gray_path = plot_grey(network_new_path)
# scatter_path = '/home/liqing/MORL/Metro-Line/Metro-Line/pic/scatter_new.png'

# network_gray_path = '/home/liqing/MORL/Metro-Line/Metro-Line/pic/network_gray.png'

# network_new_path ='/home/liqing/MORL/Metro-Line/Metro-Line/pic/network_new.png'

# verse = transPNG(scatter_path)

# plot_blend(verse, network_gray_path, network_new_path)

#plot_training(training_path)








        






















#plt.yaxis.set_major_locator(MultipleLocator(1.0))

#plt.grid(which='major',axis='both')










        