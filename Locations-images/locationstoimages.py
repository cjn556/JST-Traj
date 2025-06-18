import os
import numpy as np
import pandas as pd
import cv2
import ast


# 读取所有的txt文件
data_dir = ''  #读取文件
dirlist = os.listdir(data_dir)
file_dirs = []
for dir in dirlist:
  file_dirs.append(data_dir + '/' + dir)
data_list = []
time_list = []


B1 = 31.10, 121.30
B2 = 31.40, 121.60
num_days = 7
intervals_per_day = 288
total_intervals = num_days * intervals_per_day
output_width, output_height = 256, 256
point_size = 1
start_date = pd.to_datetime() #开始时间
end_date = pd.to_datetime() #结束时间


time_interval = 300
color_time = pd.read_csv() #读取映射表
colors = color_time['RGB'].tolist()

num = 0
for file in file_dirs:
    image = np.ones((output_width, output_height, 3), np.uint8) * 255
    data = pd.read_csv(file, index_col=0)
    if len(data) == 0:
        continue
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data['Date'] = data['Datetime'].dt.strftime('%d')
    data['TimeDelta'] = (data['Datetime'] - start_date).dt.total_seconds()
    data['TimeBin'] = (data['TimeDelta'] // time_interval).astype(int)
    data = np.array(data)
    if len(data) > 200:
        for i in range(0, len(data)):
            data_list = data[i] #获取轨迹中某一个点的经纬度
            Lo = data_list[0] #获取轨迹中某一个点的经度
            La = data_list[1] #获取轨迹中某一个点的纬度
            time = data_list[5]
            la_norm = (La - B1[0]) / (B2[0] - B1[0])
            lo_norm = (Lo - B1[1]) / (B2[1] - B1[1])
            la_pixel = int(la_norm * (output_height - 1))
            lo_pixel = int(lo_norm * (output_width - 1))
            if time >= total_intervals:
                time = total_intervals - 1
            elif time < 0:
                time = 0  # 时间段编号为负数归到第一个颜色
            color = colors[time]
            color = ast.literal_eval(color)
            for dx in range(-point_size // 2, point_size // 2 + 1):
                for dy in range(-point_size // 2, point_size // 2 + 1):
                    x, y = lo_pixel + dx, la_pixel + dy
                    if 0 <= x < output_width and 0 <= y < output_height:
                        image[y, x] = color  # 在图像上绘制点
    else:
        continue
    image2 = np.array(image)
    date = data[0][3]
    date_folder = os.path.join('', f"date_{date}")
    if not os.path.exists(date_folder):
        os.makedirs(date_folder)
    new_filename = f"{num} trajectory in Shanghai on {date}.png"
    file_path_image = os.path.join(date_folder, new_filename)
    cv2.imwrite(file_path_image, image2)
    print(f"{num} trajectory in Shanghai on {date}.png 保存成功")
    num += 1


