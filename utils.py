import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pandas import DataFrame
from cycler import cycler
from gym_auv.utils.controllers import PI, PID

PI = PI()
PID_cross = PID(Kp=1.8, Ki=0.01, Kd=0.035)
PID_cross = PID(Kp=1.8, Ki=0.01, Kd=0.035)

matplotlib.use('TkAgg')


def parse_experiment_info():
    # 解析实验信息，从命令行参数中获取实验编号、场景名称等信息，并返回实验目录路径、代理路径和场景名称。
    """Parser for the flags that can be passed with the run/train/test scripts."""
    #  创建了一个参数解析器对象
    parser = argparse.ArgumentParser()
    #  --exp_id，用于指定要运行/训练/测试的实验编号。type=int 表示参数值将被解析为整数类型，help 参数提供了参数的描述信息。
    parser.add_argument("--exp_id", type=int, help="Which experiment number to run/train/test")
    # 添加了一个命令行参数 --scenario，用于指定要运行的场景。default="expert" 表示如果未提供该参数，则默认场景为 "expert"，type=str 表示参数值将被解析为字符串类型。
    parser.add_argument("--scenario", default="beginner", type=str, help="Which scenario to run")
    parser.add_argument("--controller_scenario", default=None, type=str, help="Which scenario the agent was trained in")
    parser.add_argument("--controller", default=None, type=int, help="Which model to load as main controller. Requires only integer")
    #  解析命令行参数，并将解析结果保存在 args 变量中。
    args = parser.parse_args()

    #   根据解析得到的命令行参数构建了实验目录路径 experiment_dir 和代理路径 agent_path。
    #   具体做法是根据参数指定的实验编号、场景名称、控制器所在场景以及控制器模型编号构建相应的路径。
    experiment_dir = os.path.join(r"./log", r"Experiment {}".format(args.exp_id))

    if args.controller_scenario is not None:
        agent_path = os.path.join(experiment_dir, args.controller_scenario, "agents")
    else:
        agent_path = os.path.join(experiment_dir, args.scenario, "agents")
    if args.controller is not None:
        agent_path = os.path.join(agent_path, "model_" + str(args.controller) + ".pkl")
    else:
        agent_path = os.path.join(agent_path, "best_model")
    return experiment_dir, agent_path, args.scenario


def calculate_IAE(sim_df):
    """
    Calculates and prints the integral absolute error provided an environment id and simulation data
    """
    IAE_cross = sim_df[r"e"].abs().sum()
    IAE_vertical = sim_df[r"h"].abs().sum()
    print("IAE Cross track: {}, IAE Vertical track: {}".format(IAE_cross, IAE_vertical))
    return IAE_cross, IAE_vertical

def simulate_environment(env, agent):
    global error_labels, current_labels, input_labels, state_labels
    state_labels = [r"$N$", r"$E$", r"$D$", r"$\phi$", r"$\theta$", r"$\psi$", r"$u$", r"$v$", r"$w$", r"$p$", r"$q$", r"$r$"] # 加r防止字符转义
    current_labels = [r"$u_c$", r"$v_c$", r"$w_c$"]
    input_labels = [r"$\eta$", r"$\delta_r$", r"$\delta_s$"]
    error_labels = [r"$\tilde{u}$", r"$\tilde{\chi}$", r"e", r"$\tilde{\upsilon}$", r"h"]
    labels = np.hstack(["Time", state_labels, input_labels, error_labels, current_labels]) # 一列一列摆放
    
    done = False
    env.reset()
    while not done:
        action = agent.predict(env.observation, deterministic=True)[0]
        # action = [0,0]
        _, _, done, _ = env.step(action)
    errors = np.array(env.past_errors)
    time = np.array(env.time).reshape((env.total_t_steps,1)) # 一列
    sim_data = np.hstack([time, env.past_states, env.past_actions, errors, env.current_history])
    df = DataFrame(sim_data, columns=labels) # 一组有序的列，指定列索引的值
    error_labels = [r"e", r"h"]
    return df

def set_default_plot_rc():
    """Sets the style for the plots report-ready"""
    colors = (cycler(color= ['#EE6666', '#3388BB', '#88DD89', '#EECC55', '#88BB44', '#FFBBBB']) +
                cycler(linestyle=['-',       '-',      '-',     '--',      ':',       '-.'])) # 将两个对象的元素按次序一一组合
    plt.rc('axes', facecolor='#ffffff', edgecolor='black', # 修改全局默认属性
        axisbelow=True, grid=True, prop_cycle=colors) # 坐标轴：背景颜色、边的颜色……color cycler and a linestyle cycler
    plt.rc('xtick', direction='out', color='black', labelsize=14)
    plt.rc('ytick', direction='out', color='black', labelsize=14) # 横、纵轴刻度
    plt.rc('font', family='Times New Roman')
    plt.rc('patch', edgecolor='#ffffff') #  斑纹
    plt.rc('lines', linewidth=4) # 线条样式

def plot_attitude(sim_df):
    """Plots the state trajectories for the simulation data"""
    set_default_plot_rc() # 画图设置
    ax = sim_df.plot(x="Time", y=[r"$\phi$",r"$\theta$", r"$\psi$"], kind="line") # pandas.DataFrame.plot()。kind可以是line，bar，barth，kde
    ax.set_xlabel(xlabel="Time [s]",fontsize=14)
    ax.set_ylabel(ylabel="Angular position [rad]",fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-np.pi,np.pi])
    plt.show()

def plot_velocity(sim_df):
    """Plots the velocity trajectories for the simulation data"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$u$",r"$v$"], kind="line")
    ax.plot(sim_df["Time"], sim_df[r"$w$"], dashes=[3,3], color="#88DD89", label=r"$w$")
    ax.plot([0,sim_df["Time"].iloc[-1]], [1.0,1.0], label=r"$u_d$") # iloc从0开始计数，先选取行再选取列
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Velocity [m/s]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-0.25,2.25])
    plt.show()

def plot_angular_velocity(sim_df):
    """Plots the angular velocity trajectories for the simulation data"""
    set_default_plot_rc()
    ax = sim_df.plot(x="Time", y=[r"$p$",r"$q$", r"$r$"], kind="line")
    plt.subplots_adjust(left=0.18)
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Angular Velocity [rad/s]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    # ax.set_ylim([-1,1])
    plt.show()

def plot_control_inputs(sim_dfs):
    """ Plot control inputs from simulation data"""
    set_default_plot_rc()
    c = ['#EE6666', '#88BB44', '#EECC55']
    for i, sim_df in enumerate(sim_dfs):
        control = np.sqrt(sim_df[r"$\delta_r$"]**2+sim_df[r"$\delta_s$"]**2)
        plt.plot(sim_df["Time"], control, linewidth=4, color=c[i])
    plt.subplots_adjust(left=0.15)
    plt.xlabel(xlabel="Time [s]", fontsize=14)
    plt.ylabel(ylabel="Normalized Input", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    # plt.legend([r"$\lambda_r=0.9$", r"$\lambda_r=0.5$", r"$\lambda_r=0.1$"], loc="upper right", fontsize=14)
    # plt.ylim([-1.25,1.25])
    plt.show()

def plot_control_errors(sim_dfs):
    """
    Plot control inputs from simulation data
    """
    #error_labels = [r'e', r'h']
    set_default_plot_rc()
    c = ['#EE6666', '#88BB44', '#EECC55']
    for i, sim_df in enumerate(sim_dfs):
        error = np.sqrt(sim_df[r"e"]**2+sim_df[r"h"]**2)
        plt.plot(sim_df["Time"], error, linewidth=4, color=c[i])
    plt.xlabel(xlabel="Time [s]", fontsize=12)
    plt.ylabel(ylabel="Tracking Error [m]", fontsize=12)
    # plt.ylim([0,15])
    # plt.legend([r"$\lambda_r=0.9$", r"$\lambda_r=0.5$", r"$\lambda_r=0.1$"], loc="upper right", fontsize=14)
    plt.show()

def plot_3d(env, sim_df):
    """
    Plots the AUV path in 3D inside the environment provided.
    """
    plt.rcdefaults() # 从Matplotlib的内部默认样式中还原rc参数
    plt.rc('lines', linewidth=2)
    plt.rc('font', family='Times New Roman')
    ax = env.plot3D()#(wps_on=False)
    ax.plot3D(sim_df[r"$N$"], sim_df[r"$E$"], sim_df[r"$D$"], color="#EE6666", linestyle="dashed", label="AUV Path")#, linestyle="dashed")
    ax.set_xlabel(xlabel="North [m]", fontsize=14)
    ax.set_ylabel(ylabel="East [m]", fontsize=14)
    ax.set_zlabel(zlabel="Down [m]", fontsize=14)
    ax.legend(loc="upper right", fontsize=14)
    ax.view_init(elev=29, azim=-115)
    plt.show()

def plot_multiple_3d(env, sim_dfs):
    """
    Plots multiple AUV paths in 3D inside the environment provided.
    """
    plt.rcdefaults()
    c = ['#EE6666', '#88BB44', '#EECC55']
    styles = ["dashed", "dashed", "dashed"]
    plt.rc('lines', linewidth=3)
    ax = env.plot3D()#(wps_on=False)
    for i,sim_df in enumerate(sim_dfs):
        ax.plot3D(sim_df[r"$N$"], sim_df[r"$E$"], sim_df[r"$D$"], color=c[i], linestyle=styles[i])
    ax.set_xlabel(xlabel="North [m]", fontsize=14)
    ax.set_ylabel(ylabel="East [m]", fontsize=14)
    ax.set_zlabel(zlabel="Down [m]", fontsize=14)
    ax.legend(["Path",r"$\lambda_r=0.9$", r"$\lambda_r=0.5$",r"$\lambda_r=0.1$"], loc="upper right", fontsize=14)
    plt.show()

def plot_current_data(sim_df):
    set_default_plot_rc()
    #---------------Plot current intensity（强度）------------------------------------
    ax1 = sim_df.plot(x="Time", y=current_labels, linewidth=4, style=["-", "-", "-"] )
    ax1.set_title("Current", fontsize=18)
    ax1.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax1.set_ylabel(ylabel="Velocity [m/s]", fontsize=14)
    plt.subplots_adjust(left=0.18)
    # ax1.set_ylim([-1.25,1.25])
    ax1.legend(loc="right", fontsize=14)
    #ax1.grid(color='k', linestyle='-', linewidth=0.1)
    plt.show()
    #---------------Plot current direction------------------------------------
    """
    ax2 = ax1.twinx() # twinx()函数表示共享x轴
    ax2 = sim_df.plot(x="Time", y=[r"$\alpha_c$", r"$\beta_c$"], linewidth=4, style=["-", "--"] )
    ax2.set_title("Current", fontsize=18)
    ax2.set_xlabel(xlabel="Time [s]", fontsize=12)
    ax2.set_ylabel(ylabel="Direction [rad]", fontsize=12)
    ax2.set_ylim([-np.pi, np.pi])
    ax2.legend(loc="right", fontsize=12)
    ax2.grid(color='k', linestyle='-', linewidth=0.1)
    plt.show()
    """

#  用于绘制碰撞奖励函数图。
def plot_collision_reward_function():
    #   生成水平、垂直方向传感器角度的等间距样本数组，范围从 -70 度到 70 度，共 300 个样本。
    horizontal_angles = np.linspace(-70, 70, 300)
    vertical_angles = np.linspace(-70, 70, 300)
    gamma_x = 25
    epsilon = 0.05
    #  初始化传感器读数，生成一个形状为 (300, 300) 的数组，所有元素均为 0.4。
    sensor_readings = 0.4*np.ones((300,300))
    #  初始化图像数组，生成一个形状为 (300, 300) 的零矩阵。
    image = np.zeros((len(vertical_angles), len(horizontal_angles)))
    # 使用两层循环遍历水平角度和垂直角度，并计算对应位置的碰撞奖励函数值，存储在 image 数组中。
    for i, horizontal_angle in enumerate(horizontal_angles):
        # 水平角度和垂直角度对应的惩罚因子，这些因子的值随着角度的绝对值增大而减小。
        horizontal_factor = (1-(abs(horizontal_angle)/horizontal_angles[-1]))
        for j, vertical_angle in enumerate(vertical_angles):
            vertical_factor = (1-(abs(vertical_angle)/vertical_angles[-1]))
            beta = horizontal_factor*vertical_factor + epsilon
            image[j,i] = beta*(1/(gamma_x*(sensor_readings[j,i])**4))  #  对应位置的碰撞奖励函数值。
    print(image.round(2))  #  用于打印图像矩阵，保留两位小数，以便查看图像数据
    ax = plt.axes() # 设置具体某一个坐标轴的属性
    plt.colorbar(plt.imshow(image),ax=ax) # 在图像上添加一个色度条，色度条的值范围对应图像的取值范围。
    ax.imshow(image, extent=[-70,70,-70,70]) # 在坐标轴上绘制图像，使用 extent 参数指定水平和垂直方向的坐标范围。
    ax.set_ylabel("Vertical vessel-relative sensor angle [deg]", fontsize=14)
    ax.set_xlabel("Horizontal vessel-relative sensor angle [deg]", fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14)  #   设置 x 轴刻度的字体大小
    ax.yaxis.set_tick_params(labelsize=14)
    plt.show()


if __name__ == "__main__":
    plot_collision_reward_function()
