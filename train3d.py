import os
import gym
import torch
import gym_auv
import stable_baselines3.common.results_plotter as results_plotter
import numpy as np
# import tensorflow as tf

from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.policies import MlpPolicy, LstmPolicy
# from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO2
from stable_baselines3.ppo2.policies import ModelPolicy
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.schedules import LinearSchedule
from utils import parse_experiment_info

#  ADD
from typing import Tuple

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # #屏蔽通知信息、警告信息和报错信（INFO\WARNING\FATAL）
torch.set_num_threads(1) # 设置pytorch进行CPU多线程并行计算时所占用的线程数
# scenarios = ["beginner", "intermediate", "proficient", "advanced", "expert"]

'''这个scenarios里面都没有expert，为什么utils.py里面Line26默认使用expert还不报错的'''

scenarios = ["beginner", "intermediate"]
# 定义了一系列超参数，如步数 (n_steps)、批大小 (batch_size)、学习率 (learning_rate
# hyperparams = {
#     'n_steps': 1024,
#     'batch_size': 256,
#     'learning_rate': 1e-5,
#     # 'batch_size': 32,
#     'gae_lambda': 0.95,
#     'gamma': 0.99,
#     'n_epochs': 4,
#     'clip_range': 0.2,
#     'ent_coef': 0.01,
#     'verbose': 2,
#     # 'seed': 1
#     }

# #  0425night
# hyperparams = {
#     'n_steps': 2048,
#     'batch_size': 128,
#     'learning_rate': 2e-5,
#     # 'batch_size': 32,
#     'gae_lambda': 0.95,
#     'gamma': 0.99,
#     'n_epochs': 4,
#     'clip_range': 0.2,
#     'ent_coef': 0.01,  # 熵系数，可以增加探索。如果训练过程过于震荡，可以尝试减少熵系数以减少策略变化的幅度。
#     'verbose': 2,
#     # 'seed': 1
#     }

# 0427 night
# hyperparams = {
#     'n_steps': 2048,
#     'batch_size': 256,
#     'learning_rate': 2e-4,
#     # 'batch_size': 32,
#     'gae_lambda': 0.95,
#     'gamma': 0.99,
#     'n_epochs': 8,
#     'clip_range': 0.2,  # 裁剪范围用于限制策略更新的幅度，以避免过大的策略更新导致的训练不稳定
#     'ent_coef': 0.01,  # 熵系数，可以增加探索。如果训练过程过于震荡，可以尝试减少熵系数以减少策略变化的幅度。
#     'verbose': 2,
#     # 'seed': 1
#     }

#  0428
hyperparams = {
    'n_steps': 2048,
    'batch_size': 128,
    'learning_rate': 1e-4,
    # 'batch_size': 32,
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'n_epochs': 4,
    'clip_range': 0.25,
    'ent_coef': 0.01,
    'verbose': 2,
    # 'seed': 1
    }

#  回调函数   用作训练时的回调。在每个步骤结束时，它检查是否需要保存模型。
def callback(_locals, _globals):
    global n_steps, best_mean_reward
    if (n_steps + 1) % 5 == 0:
        _locals['self'].save(os.path.join(agents_dir, "model_" + str(n_steps+1) + ".pkl"))
    n_steps += 1
    return True


#  添加模块去保存注意力机制的参数
# class CustomPolicy(ModelPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomPolicy, self).__init__(*args, **kwargs)
#         input_dim = 64  # 替换为你的输入维度
#         attention_heads = 1  # 替换为你的注意力头数
#         attention_dim = 32  # 替换为你的注意力维度
#         # 在初始化中定义注意力模块的参数
#         self.attention_module = SelfAttention(input_dim, attention_heads, attention_dim)
#
#         # 将 SelfAttention 模块添加为子模块
#         self.add_module("attention_module", self.attention_module)

    # def forward(self, obs: torch.Tensor, deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     # 在前向传播中应用注意力模块
    #     obs = self.attention_module(obs)
    #     return super(CustomPolicy, self).forward(obs, deterministic)



if __name__ == '__main__':
    #  第一个值（实验目录的路径）赋值给 experiment_dir 变量，后续的两个值忽略
    experiment_dir, _, _ = parse_experiment_info()
    
    # for i, scen in enumerate(scenarios):  #  循环遍历每个任务场景
    scen = scenarios[1]
    #   experiment_dir 是实验目录的路径，scen 是当前场景的名称，"agents" 是存储代理模型的子目录名称。
    #  结果会将这三个部分连接起来形成完整的路径，存储在 agents_dir 变量中。
    agents_dir = os.path.join(experiment_dir, scen, "agents")
    #  创建了另一个路径，用于存储 TensorBoard 日志文件。它的结构类似于代理模型的路径
    tensorboard_dir = os.path.join(experiment_dir, scen, "tensorboard")
    #  用于创建代理模型和 TensorBoard 日志的目录
    os.makedirs(agents_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    #将 TensorBoard 日志目录的路径更新到超参数字典 hyperparams 中的 tensorboard_log 键对应的值上
    #这样做的目的是，后续训练过程中，代理将日志信息写入到这个目录下，便于后续使用 TensorBoard 进行可视化分析。
    hyperparams["tensorboard_log"] = tensorboard_dir

    # 定义环境数量
    num_envs = 1  # num_env=4 => 4 processes
    # 如果环境数量大于 1，则使用 SubprocVecEnv 创建多个并行的环境实例。
    # 如果环境数量为 1，则使用 DummyVecEnv 创建单个环境实例。
    if num_envs > 1:
        env = SubprocVecEnv([lambda: Monitor(gym.make("PathColav3d-v0", scenario=scen), agents_dir, allow_early_resets=True) for i in range(num_envs)])
    else:
        env = DummyVecEnv([lambda: Monitor(gym.make("PathColav3d-v0", scenario=scen), agents_dir, allow_early_resets=True)])
    # hyperparams["batch_size"] = (hyperparams["n_steps"]*num_envs)//32 # 4个是128，1个是32

    if scen == "beginner":
        # 如果当前场景是 "beginner"，则使用 PPO2 算法的构造函数创建新的代理(agent)，并使用指定的策略模型 ModelPolicy
        # continue
        agent = PPO2(ModelPolicy, env, **hyperparams) # ActorCriticPolicy，默认网络pi=[64, 64], vf=[64, 64]
        print("beginner当前执行的是beginner模式")
    else:
        # 否则，加载上一个场景的最佳模型，继续在当前环境上进行训练。加载模型后，调用 _setup_model() 方法来设置代理的模型。
        print("intermediate当前执行的是intermediate模式")
        continual_model = os.path.join(experiment_dir, scenarios[0], "agents", "best_model")
        agent = PPO2.load(continual_model, env=env, **hyperparams)
        agent._setup_model()

        # 创建一个 EvalCallback 对象，用于在训练过程中定期评估代理的性能。这将在每 500 个训练步骤后进行评估。
        eval_callback = EvalCallback(env, best_model_save_path=agents_dir,
                             log_path=agents_dir, eval_freq=500,
                             deterministic=True, render=False)

        #   best_mean_reward：代表了在训练过程中的最佳平均奖励值。被初始化为负无穷大 (-np.inf)。在训练过程中，如果出现了新的平均奖励值高于当前的 best_mean_reward，则 best_mean_reward 会被更新。
        #  n_steps：代表了当前的训练步数。在这里，它被初始化为0，表示训练刚开始。
        #  timesteps：代表了总的训练步数。300e3 表示300000步，i*150e3 表示每个场景的训练步数增量，其中 i 是场景的索引，乘以 150e3 是为了让每个场景的训练步数逐渐增加。
        best_mean_reward, n_steps, timesteps = -np.inf, 0, int(400e3) # 300*1000+i*150*1000
        # learn() 方法，用于执行强化学习算法的训练。callback该回调函数用于在每个训练步骤结束时进行评估。
        agent.learn(total_timesteps=timesteps, tb_log_name="PPO2", callback=eval_callback) # Callback that will be called at each step每个时间步，而不是每次learn被调用
        #  这一行代码用于生成最终模型保存的路径
        save_path = os.path.join(agents_dir, "last_model.pkl")
        #  调用了代理(agent)的 save() 方法，用于将训练得到的最终模型保存到磁盘上。
        # 参数 save_path 指定了保存模型的路径。
        agent.save(save_path)