'''
    路径规划、控制器设计以及环境仿真等功能
'''
import numpy as np
import gym
import gym_auv.utils.geomutils as geom
import matplotlib.pyplot as plt
import skimage.measure

from gym_auv.objects.auv3d import AUV3D
from gym_auv.objects.current3d import Current
from gym_auv.objects.QPMI import QPMI, generate_random_waypoints # QPMI插值方法
from gym_auv.objects.path3d import Path3D
from gym_auv.objects.obstacle3d import Obstacle
from gym_auv.utils.controllers import PI, PID

#  测试用的路径点
#    包含了十个路径点，这些点描述了一个曲线路径，大致呈现出一个闭合的环路。
test_waypoints = np.array([np.array([0,0,0]), np.array([20,10,15]), np.array([50,20,20]), np.array([80,20,40]), np.array([90,50,50]),
                           np.array([80,80,60]), np.array([50,80,20]), np.array([20,60,15]), np.array([20,40,10]), np.array([0,0,0])])
#    包含了五个路径点，这些点描述了另一个路径，形成了一个简单的线性路径。
test_waypoints = np.array([np.array([0,0,0]), np.array([50,15,5]), np.array([80,5,-5]), np.array([120,10,0]), np.array([150,0,0])])

#  实现了 OpenAI Gym 的接口。这个环境包括了初始化方法、重置方法、仿真步进方法等。
class PathColav3d(gym.Env):
    """
    Creates an environment with a vessel, a path and obstacles.
    """
    def __init__(self, env_config, scenario="intermediate"):
        for key in env_config:
            setattr(self, key, env_config[key]) # 设置属性值
        #  计算状态观测的维度，它等于状态观测数目、错误观测数目和输入观测数目之和。
        self.n_observations = self.n_obs_states + self.n_obs_errors + self.n_obs_inputs# + self.sensor_input_size[0]*self.sensor_input_size[1] # 状态维数
        #  创建了一个动作空间，表示机器人或车辆可执行的动作范围。
        #  这里使用了 Gym 库中的 Box 空间，它代表一个n维的箱子（矩形）空间。
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]),
                                           high=np.array([1]*self.n_actuators),
                                           dtype=np.float32) # The Box space represents an n-dimensional box
        #  创建了一个状态观测空间，表示机器人或车辆的感知状态的范围。
        #  这里同样使用了 Gym 库中的 Box 空间。
        self.observation_space = gym.spaces.Box(low=np.array([-1]*self.n_observations), # 不考虑避障不要雷达
                                                high=np.array([1]*self.n_observations),
                                                dtype=np.float32)
        
        self.scenario = scenario
        #  包含了不同场景对应的方法。
        #  它将字符串键映射到对应的场景方法。
        self.scenario_switch = {
            # Training scenarios
            "beginner": self.scenario_beginner,
            "intermediate": self.scenario_intermediate,
            # Testing scenarios
            "test_path": self.scenario_test_path,
            "test_path_current": self.scenario_test_path_current,
        }

        self.reset()

    #  重置环境状态，包括初始化AUV、路径等。
    def reset(self):
        """
        Resets environment to initial state. 
        """
        self.vessel = None
        self.path = None
        self.u_error = None
        self.e = None
        self.h = None
        self.chi_error = None
        self.upsilon_error = None
        self.waypoint_index = 0
        self.prog = 0
        self.path_prog = []
        self.success = False

        self.obstacles = []
        self.nearby_obstacles = []
        self.sensor_readings = np.zeros(shape=self.sensor_suite, dtype=float)
        self.collided = False
        self.penalize_control = 0.0

        self.observation = None
        self.action_derivative = np.zeros(self.n_actuators)
        self.past_states = []
        self.past_actions = []
        self.past_errors = []
        self.past_obs = []
        self.current_history = []
        self.time = []
        self.total_t_steps = 0
        self.reward = 0

        self.generate_environment()
        self.update_control_errors()
        self.observation = self.observe(np.zeros(6, dtype=float))
        return self.observation

    #  用于创建船舶、可能的海洋流以及一个三维路径。
    def generate_environment(self):
        """
        Generates environment with a vessel, potentially ocean current and a 3D path.
        """     
        # Generate training/test scenario
        scenario = self.scenario_switch.get(self.scenario, lambda: print("Invalid scenario")) # lambda定义了一个匿名函数
        #  调用了获取到的场景方法，得到了初始状态。
        init_state = scenario()

        # Generate AUV
        self.vessel = AUV3D(self.step_size, init_state)
        #  创建了一个比例积分（PI）控制器对象，该对象被赋值给 self.thrust_controller 属性。
        self.thrust_controller = PI()
    
    #  将3D环境的一个截面可视化
    def plot_section3(self):
        plt.rc('lines', linewidth=3)
        ax = self.plot3D(wps_on=False) # Returns 3D plot of path and obstacles.调用 plot3D 方法绘制3D图，并关闭路径点的绘制。
        ax.set_xlabel(xlabel="North [m]", fontsize=14)
        ax.set_ylabel(ylabel="East [m]", fontsize=14)
        ax.set_zlabel(zlabel="Down [m]", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([-50, 0, 50])
        ax.set_zticks([-50, 0, 50])
        ax.view_init(elev=-165, azim=-35) # 视角转换
        #  在图中标记初始位置：通过在初始位置绘制黄色的散点来标记AUV的初始位置。
        ax.scatter3D(*self.vessel.position, label="Initial Position", color="y") # 散点图

        self.axis_equal3d(ax) #  Shifts axis in 3D plots to be equal.
        ax.legend(fontsize=14)
        plt.show()
        # a.asjdl

    #  模拟环境中的一个时间步，包括更新当前状态、执行动作、计算奖励等。
    def step(self, action):
        """
        Simulates the environment one time-step. 
        """
        # Simulate Current
        self.current.sim()
        #  计算AUV在当前洋流条件下的相对洋流速度nu_c。
        nu_c = self.current(self.vessel.state)
        self.current_history.append(nu_c[0:3])

        # Simulate AUV dynamics one time-step and save action and state
        # cruise speed error, tracking errors, course and elecation errors
        # 更新控制误差，包括巡航速度误差、跟踪误差、航向误差和俯仰误差。
        self.update_control_errors()
        #  使用控制器计算推力控制器输出thrust
        thrust = self.thrust_controller.u(self.u_error)  # PI
        #  将动作和推力合并，然后进行剪裁，以确保在合理范围内。
        action = np.hstack((thrust, action))
        action = np.clip(action, np.array([0, -1, -1]), np.array([1, 1, 1]))
        #  如果过去有动作记录，计算动作的导数。
        if len(self.past_actions) > 0:
            self.action_derivative = (action[1:] - self.past_actions[-1][1:]) / (self.step_size)

        #  调用AUV对象的step方法，传递动作和相对洋流速度，更新AUV状态。
        self.vessel.step(action, nu_c)  # AUV3D
        #  将当前状态、控制误差和动作添加到历史记录中。
        self.past_states.append(np.copy(self.vessel.state))
        self.past_errors.append(np.array([self.u_error, self.chi_error, self.e, self.upsilon_error, self.h]))
        self.past_actions.append(self.vessel.input)

        #  如果存在路径，则计算AUV当前位置在路径上的最近点，并记录到self.prog中。
        if self.path:
            self.prog = self.path.get_closest_u(self.vessel.position, self.waypoint_index)  # 轨迹上离AUV最近的点
            self.path_prog.append(self.prog)

            # Check if a waypoint is passed
            #  检查是否通过了航点，如果通过了航点，则更新self.waypoint_index
            k = self.path.get_u_index(self.prog)
            if k > self.waypoint_index:
                print("Passed waypoint {:d}".format(k + 1))
                self.waypoint_index = k

        # Calculate reward based on observation and actions
        #  根据观测和动作计算当前时间步的奖励。
        # Calculates the reward function for one time step. Also checks if the episode should end.
        done, step_reward = self.step_reward(self.observation,
                                             action)

        info = {}

        # Make next observation  根据当前洋流条件更新观测。
        self.observation = self.observe(nu_c)
        self.past_obs.append(self.observation)

        # Save sim time info
        self.total_t_steps += 1
        self.time.append(self.total_t_steps * self.step_size)

        return self.observation, step_reward, done, info

    #  获取环境的观测信息。
    #  线速度；横滚、俯仰和航向角；角速度；洋流速度；航向、仰角误差
    def observe(self, nu_c):
        """
        Returns observations of the environment. 
        """
        obs = np.zeros((self.n_observations,)) # 状态是运用运动学和动力学公式算出来的
        #  AUV相对洋流的线速度在三个轴上的比例，并进行剪裁，以确保在范围[-1, 1]内
        obs[0] = np.clip(self.vessel.relative_velocity[0] / 2, -1, 1) # 线速度，标准化
        obs[1] = np.clip(self.vessel.relative_velocity[1] / 0.3, -1, 1)
        obs[2] = np.clip(self.vessel.relative_velocity[2] / 0.3, -1, 1)
        #  计算AUV的横滚、俯仰和航向角，并进行剪裁，以确保在范围[-1, 1]内
        obs[3] = np.clip(self.vessel.roll / np.pi, -1, 1)  #  横滚角
        obs[4] = np.clip(self.vessel.pitch / np.pi, -1, 1)  #  俯仰角
        obs[5] = np.clip(self.vessel.heading / np.pi, -1, 1)  #  航向角
        #  角速度在三个轴上的比例，并进行剪裁，以确保在范围[-1, 1]内
        obs[6] = np.clip(self.vessel.angular_velocity[0] / 1.2, -1, 1) # 角速度
        obs[7] = np.clip(self.vessel.angular_velocity[1] / 0.4, -1, 1)
        obs[8] = np.clip(self.vessel.angular_velocity[2] / 0.4, -1, 1)
        #  洋流速度
        obs[9] = np.clip(nu_c[0] / 1, -1, 1) # Ocean current velocity
        obs[10] = np.clip(nu_c[1] / 1, -1, 1)
        obs[11] = np.clip(nu_c[2] / 1, -1, 1)

        obs[12] = self.chi_error # Course error  航向误差
        obs[13] = self.upsilon_error # Elevation error  仰角误差

        # Update nearby obstacles and calculate distances
        #if self.total_t_steps % self.update_sensor_step == 0:
            #self.update_nearby_obstacles() # Updates the nearby_obstacles array
            #self.update_sensor_readings() # Updates the sonar data closeness array
            #self.sonar_observations = skimage.measure.block_reduce(self.sensor_readings, (2,2), np.max)
            #self.update_sensor_readings_with_plots() #(Debugging)
        #obs[14:] = self.sonar_observations.flatten()
        return obs

    #  计算每个时间步的奖励，并判断是否终止当前 时间步
    #  分别计算横滚角、横滚角速度奖励；舵角和升降舵角奖励；航向误差和仰角误差奖励  @@@ 加权求和，并将当前时间步奖励加到总奖励中
    #  根据四个条件，判断是否应该结束该时间步
    #  返回done和这个时间步的奖励
    def step_reward(self, obs, action):
        """
        Calculates the reward function for one time step. Also checks if the episode should end. 
        """
        done = False
        step_reward = 0

        #  根据横滚角和横滚角速度计算的奖励。
        reward_roll = self.vessel.roll ** 2 * self.reward_roll + self.vessel.angular_velocity[
            0] ** 2 * self.reward_rollrate
        #  根据舵角和升降舵角控制命令的平方和计算的奖励。
        reward_control = action[1] ** 2 * self.reward_use_rudder + action[2] ** 2 * self.reward_use_elevator
        #  根据航向误差和仰角误差的平方和计算的奖励。
        reward_path_following = self.chi_error ** 2 * self.reward_heading_error + self.upsilon_error ** 2 * self.reward_pitch_error
        #  将以上三个奖励加权求和，并乘以权重系数lambda_reward。
        step_reward = self.lambda_reward * reward_path_following + reward_roll + reward_control
        #  将当前时间步的奖励加到总奖励中。
        self.reward += step_reward

        #  根据以下条件之一，判断是否应该结束该时间步：
        # 累积奖励小于最小奖励阈值 min_reward。
        # 达到最大时间步数 max_t_steps。
        # AUV到达路径终点，且距离终点的距离小于接受半径 accept_rad，且当前航点是倒数第二个航点。
        # AUV距离路径终点的剩余距离小于接受半径的一半。
        end_cond_1 = self.reward < self.min_reward
        end_cond_2 = self.total_t_steps >= self.max_t_steps  # 达到最大步数
        end_cond_3 = np.linalg.norm(
            self.path.get_endpoint() - self.vessel.position) < self.accept_rad and self.waypoint_index == self.n_waypoints - 2  # np.linalg.norm求范数，已到终点
        end_cond_4 = abs(self.prog - self.path.length) <= self.accept_rad / 2.0

        if end_cond_1 or end_cond_2 or end_cond_3 or end_cond_4:
            if end_cond_3:
                print("AUV reached target!")
                self.success = True
            elif self.collided:
                print("AUV collided!")
                print(np.round(self.sensor_readings, 2))
                self.success = False
            print(
                "Episode finished after {} timesteps with reward: {}".format(self.total_t_steps, self.reward.round(1)))
            done = True
        return done, step_reward

    #  更新控制器误差，包括巡航速度误差、航向误差和俯仰误差。
    def update_control_errors(self):
        # Update cruise speed error  更新巡航速度误差
        #  计算AUV当前的相对线速度与目标巡航速度之间的差值，并进行剪裁，以确保在范围[-1, 1]内。
        self.u_error = np.clip((self.cruise_speed - self.vessel.relative_velocity[0])/2, -1, 1)
        self.chi_error = 0.0 # heading_error  航向误差
        self.e = 0.0
        self.upsilon_error = 0.0 # pitch error  俯仰误差
        self.h = 0.0

        # Get path course航向 and elevation仰角
        s = self.prog
        #  获取当前路径点处的航向角chi_p和俯仰角upsilon_p
        chi_p, upsilon_p = self.path.get_direction_angles(s)

        # Calculate tracking errors  跟踪误差
        SF_rotation = geom.Rzyx(0,upsilon_p,chi_p) # xyz三个方位角的正弦余弦
        #  计算特定坐标系下的误差向量epsilon，其为目标路径点和AUV位置之间的差向量经过路径坐标系的旋转。
        epsilon = np.transpose(SF_rotation).dot(self.vessel.position-self.path(self.prog)) # transpose矩阵转置
        #  提取误差向量的横向误差e和垂直误差h。
        e = epsilon[1]
        h = epsilon[2]

        # Calculate course and elevation errors from tracking errors
        #  路径偏差角chi_r和仰角偏差角upsilon_r
        chi_r = np.arctan2(-e, self.la_dist)
        upsilon_r = np.arctan2(h, np.sqrt(e**2 + self.la_dist**2))
        chi_d = chi_p + chi_r
        upsilon_d = upsilon_p + upsilon_r
        self.chi_error = np.clip(geom.ssa(self.vessel.chi - chi_d)/np.pi, -1, 1)
        #self.e = np.clip(e/12, -1, 1)
        self.e = e
        self.upsilon_error = np.clip(geom.ssa(self.vessel.upsilon - upsilon_d)/np.pi, -1, 1)
        #self.h = np.clip(h/12, -1, 1)
        self.h = h

    #  绘制环境的3D图像
    def plot3D(self, wps_on=True):
        """
        Returns 3D plot of path and obstacles.
        """
        ax = self.path.plot_path(wps_on)
        for obstacle in self.obstacles:     #  使用一个循环遍历 self.obstacles 列表中的每个障碍物
            ax.plot_surface(*obstacle.return_plot_variables(), color='r')
            #对于每个障碍物，调用其 return_plot_variables() 方法获取绘图参数，
            # 并使用 ax.plot_surface() 方法绘制障碍物的表面。
        return self.axis_equal3d(ax) # Shifts axis in 3D plots to be equal
        #  将3D图中的轴调整为相等，使得三个轴的比例相同。
        # 返回调整后的轴对象 ax。

    #  调整三维图中的坐标轴，使它们具有相等的比例
    def axis_equal3d(self, ax):
        """
        Shifts axis in 3D plots to be equal. Especially useful when plotting obstacles, so they appear spherical(圆滑).
        
        Parameters:
        ----------
        ax : matplotlib.axes
            The axes to be shifted. 
        """
        #  使用了列表推导式和 getattr() 函数来获取三个坐标轴的范围。
        #  getattr(ax, 'get_{}lim'.format(dim))() 用于获取 ax 对象中每个维度的上下限。
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']) # 获取三个坐标轴
        #  计算每个维度上的范围。
        sz = extents[:,1] - extents[:,0]
        #  计算每个维度的中心点。
        centers = np.mean(extents, axis=1)
        #  计算三个维度上范围的最大值。
        maxsize = max(abs(sz))
        #  计算半径，即范围最大值的一半。
        r = maxsize/2
        #  使用 getattr() 函数动态地调用 ax 对象的 set_xlim、set_ylim 和 set_zlim 方法，分别设置x、y和z轴的范围。
        #  范围设置为中心点减去半径到中心点加上半径。
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        return ax

    #  设置一个初学者级别的环境场景
    def scenario_beginner(self):
        initial_state = np.zeros(6)
        #  表示当前环境的洋流情况
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0) #Current object with zero velocity
        #  生成一系列随机航点 waypoints，这些航点将用于生成轨迹。
        waypoints = generate_random_waypoints(self.n_waypoints)
        self.path = QPMI(waypoints)  #  使用生成的随机航点来插值生成AUV的轨迹。
        init_pos = [np.random.uniform(0,2)*(-5), np.random.normal(0,1)*5, np.random.normal(0,1)*5] # 均匀分布，正态分布
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state

    #  提供一个中等难度的环境设置，洋流速度的增加
    def scenario_intermediate(self):
        #  调用 scenario_beginner 方法获取初学者级别场景的初始状态。
        initial_state = self.scenario_beginner()
        self.current = Current(mu=0.2, Vmin=0.2, Vmax=1, Vc_init=np.random.uniform(0.2, 1),
                                    alpha_init=np.random.uniform(-np.pi, np.pi), beta_init=np.random.uniform(-np.pi/4, np.pi/4), t_step=self.step_size)
        #  用于控制处罚
        self.penalize_control = 1.0
        return initial_state

    #  测试AUV在给定测试轨迹下的行为。
    def scenario_test_path(self):
        self.n_waypoints = len(test_waypoints)
        self.path = QPMI(test_waypoints)
        self.current = Current(mu=0, Vmin=0, Vmax=0, Vc_init=0, alpha_init=0, beta_init=0, t_step=0)
        init_pos = [0,0,0]
        # init_pos = [np.random.uniform(0, 2) * (-5), np.random.normal(0, 1) * 5, np.random.normal(0, 1) * 5]  # 均匀分布，正态分布
        init_attitude = np.array([0, self.path.get_direction_angles(0)[1], self.path.get_direction_angles(0)[0]])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state
        
    #  提供一个在测试轨迹上存在固定洋流影响的环境设置
    def scenario_test_path_current(self):
        initial_state = self.scenario_test_path()
        self.current = Current(mu=0, Vmin=0.75, Vmax=0.75, Vc_init=0.75, alpha_init=np.pi/4, beta_init=np.pi/6, t_step=0)
        return initial_state