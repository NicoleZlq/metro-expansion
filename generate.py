import random

# 定义生成随机数的函数
def generate_random_weight():
    # 80%的概率生成500-10000之间的数
    if random.random() < 0.99:
        return random.randint(1, 5)
    else:
        return random.randint(70000,90000)

# 读取原始文件并生成新的权重
def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            index1, index2, _ = line.rstrip().split('\t')
            a = int(index1)
            b = int(index2)
            if  361<a<422:
                index1 = 40
            if 440<b<550:
                index2 = 120
            
            file.write(f"{index1}\t{index2}\t{_}\n")

# 指定输入和输出文件
input_file_path = 'E:/UMstudy/Paper manu/part C/Metro-Line/od_index.txt'
output_file_path = 'E:/UMstudy/Paper manu/part C/Metro-Line/od_index_dyna.txt'

# 调用函数处理文件
process_file(input_file_path, output_file_path)