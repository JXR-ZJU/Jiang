import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_auv
import os

from gym_auv.utils.controllers import PI, PID
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO2
from utils import *

from gym_auv.objects.QPMI import QPMI, generate_random_waypoints
# from gym_auv.envs.pathcolav3d import
# from train3d import CustomPolicy


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 




if __name__ == "__main__":
    experiment_dir, agent_path, scenario = parse_experiment_info() # 读取命令行参数
    env = gym.make("PathColav3d-v0", scenario=scenario)
    agent = PPO2.load(agent_path)
    sim_df = simulate_environment(env, agent) # 运行仿真环境并记录
    #  将 DataFrame 对象 sim_df 中的数据保存到 CSV 文件中，文件名为 'simdata.csv'
    sim_df.to_csv(r'simdata.csv') # dataframe→csv

    # waypoints = generate_random_waypoints(env.n_waypoints)
    # QPMI.plot_path(wps_on=True)

    #  计算并打印模拟数据的积分绝对误差（Integral Absolute Error，IAE）
    #  积分绝对误差越小，说明系统的性能越好
    calculate_IAE(sim_df)
    #  生成了一个描述模拟数据中的角度位置随时间变化的图形  pitch roll yaw
    plot_attitude(sim_df)
    #  绘制模拟数据的速度轨迹图
    #  水平速度(u)和侧向速度(v),虚线来表示垂直速度(w),直线来表示目标速度(u_d)
    plot_velocity(sim_df)
    #  角速度轨迹图
    #  绕 x 轴的滚转速率(p)，绕 y 轴的俯仰速率(q)，以及绕 z 轴的偏航速率(r)
    plot_angular_velocity(sim_df)
    #  不同模拟数据集中控制输入变化的方法，有助于比较和分析不同模拟条件下的控制效果
    plot_control_inputs([sim_df])
    #  绘制模拟数据中的控制、跟踪误差轨迹图
    plot_control_errors([sim_df])
    #  路径
    plot_3d(env, sim_df)
    #  流速信息
    plot_current_data(sim_df)

