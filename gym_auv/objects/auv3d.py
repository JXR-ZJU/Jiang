import numpy as np
import gym_auv.utils.state_space_3d as ss
import gym_auv.utils.geomutils as geom
import matplotlib.pyplot as plt

from gym_auv.objects.current3d import Current
from mpl_toolkits.mplot3d import Axes3D


def odesolver45(f, y, h, nu_c): # nu_c海流速度
    """Calculate the next step of an IVP of a time-invariant ODE with a RHS(right hand side)
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 4 approx. 大约
        w: float. Order 5 approx.
    """
    s1 = f(y, nu_c) # 矩阵
    s2 = f(y + h*s1/4.0, nu_c)
    s3 = f(y + 3.0*h*s1/32.0 + 9.0*h*s2/32.0, nu_c)
    s4 = f(y + 1932.0*h*s1/2197.0 - 7200.0*h*s2/2197.0 + 7296.0*h*s3/2197.0, nu_c)
    s5 = f(y + 439.0*h*s1/216.0 - 8.0*h*s2 + 3680.0*h*s3/513.0 - 845.0*h*s4/4104.0, nu_c)
    s6 = f(y - 8.0*h*s1/27.0 + 2*h*s2 - 3544.0*h*s3/2565 + 1859.0*h*s4/4104.0 - 11.0*h*s5/40.0, nu_c)
    w = y  +  h*(25.0*s1/216.0 + 1408.0*s3/2565.0 + 2197.0*s4/4104.0 - s5/5.0)
    q = y  +  h*(16.0*s1/135.0 + 6656.0*s3/12825.0 + 28561.0*s4/56430.0 - 9.0*s5/50.0 + 2.0*s6/55.0)
    return w, q


"""
def odesolver45(f, y, h, nu_c):
    """"""Calculate the next step of an IVP of a time-invariant ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 4 approx.
        w: float. Order 5 approx.
    """"""
    s1 = f(y, nu_c)
    s2 = f(y + h*s1/5.0, nu_c)
    s3 = f(y + 3.0*h*s1/40.0 + 9.0*h*s2/40.0, nu_c)
    s4 = f(y + 44.0*h*s1/45.0 - 56.0*h*s2/15.0 + 32.0*h*s3/9.0, nu_c)
    s5 = f(y + 19372.0*h*s1/6561.0 - 25360.0*h*s2/2187.0 + 64448.0*h*s3/6561.0 - 212.0*h*s4/729.0, nu_c)
    s6 = f(y + 9017.0*h*s1/3168.0 - 355.0*h*s2/33.0 + 46732.0*h*s3/5247.0 + 49.0*h*s4/176.0 - 5103.0*h*s5/18656.0, nu_c)
    w = y  +  h*(35.0*s1/384.0 + 500.0*s3/1113.0 + 125.0*s4/192.0 - -2187.0*s5/6784.0 + 11.0*s6/84.0)
    q = y  +  h*(16.0*s1/135.0 + 6656.0*s3/12825.0 + 28561.0*s4/56430.0 - 9.0*s5/50.0 + 2.0*s6/55.0)
    return w, q
"""


class AUV3D():
    """
    Implementation of AUV dynamics. 
    """
    def __init__(self, step_size, init_eta, safety_radius=1):  # init_eta是场景
        self.state = np.hstack([init_eta, np.zeros((6,))])  # 前六个位置，后六个速度
        self.step_size = step_size   #  时间步长
        #self.alpha = self.step_size/(self.step_size + 1)
        self.alpha = self.step_size/(self.step_size + 0.2)  #  低通滤波器参数。
        self.input = np.zeros(3)  # 螺旋桨推力，舵和升降翼输入
        self.position_dot = np.zeros(3)  #  位置的导数。
        self.safety_radius = safety_radius
        self.safety_radius = 1

    #  用于执行一步动作，更新AUV的状态。
    def step(self, action, nu_c):
        #  保存上一步动作中的舵角和升降舵角
        prev_rudder_angle = self.input[1] # 上一次输入的推力，水平偏移角度和俯仰角
        prev_elevator_angle = self.input[2]

        # Un-normalize actions from neural network
        #  将动作中的推力转化为实际推力的比例。
        thrust = _surge(action[0]) # action[0]是推力占最大推力的比例
        #  将动作中的舵角和升降舵角转化为实际舵角和升降舵角的命令。
        commanded_rudder = _steer(action[1]) # 将action转化为推力和rudder、elevator的命令
        commanded_elevator = _steer(action[2])
        # Lowpass filter the rudder and elevator
        #  使用低通滤波器对当前动作和上一步动作的舵角和升降舵角进行平滑处理，以减少变化速率。
        rudder_angle = self.alpha*commanded_rudder + (1-self.alpha)*prev_rudder_angle 
        elevator_angle = self.alpha*commanded_elevator + (1-self.alpha)*prev_elevator_angle

        #self.input = np.array([thrust, commanded_rudder, commanded_elevator])
        self.input = np.array([thrust, rudder_angle, elevator_angle])
        self._sim(nu_c) # 更新状态

    #  根据给定的相对洋流速度（nu_c），使用ODE求解器计算状态变化率，并更新状态。
    # 对方位角进行限制。
    # 计算位置的导数。
    def _sim(self, nu_c): # 
        #self.state += self.state_dot(nu_c)*self.step_size
        w, q = odesolver45(self.state_dot, self.state, self.step_size, nu_c) # 运用状态变化率算出下一状态
        #self.state = q
        self.state = w # 更新状态
        self.state[3] = geom.ssa(self.state[3]) # 限制范围
        self.state[4] = geom.ssa(self.state[4])
        self.state[5] = geom.ssa(self.state[5])
        self.position_dot = self.state_dot(self.state, nu_c)[0:3] # NED下的线速度（位置的导数）

    #  计算AUV动力学方程的右手边。
    # 计算位置和速度的导数。
    def state_dot(self, state, nu_c): # 状态变化率（位置的导数，方位的导数，速度的导数）
        """
        The right hand side of the 12 ODEs governing the AUV dyanmics.
        """
        eta = self.state[:6] # 前闭后开，场景，世界坐标系中位置和方位
        nu_r = self.state[6:] # 后面6个，本体坐标系中相对于洋流速度的速度

        eta_dot = geom.J(eta).dot(nu_r+nu_c) # 坐标变换矩阵，转换到NED坐标系下的线速度和角速度，实际速度；(6,6)*(6,1)→(6,1)
        nu_r_dot = ss.M_inv().dot( # Mass Force
            ss.B(nu_r).dot(self.input) # Control Inputs
            - ss.D(nu_r).dot(nu_r) # Damping Forces
            - ss.C(nu_r).dot(nu_r) # 科氏力
            - ss.G(eta)) # Restoring Forces;(6,6)*(6,1)→(6,1)
        state_dot = np.hstack([eta_dot, nu_r_dot]) # (12,1)
        return state_dot

    @property
    def position(self):
        """
        Returns an array holding the position of the AUV in NED
        coordinates.
        """
        return self.state[0:3]


    @property
    def attitude(self):
        """
        Returns an array holding the attitude of the AUV wrt. to NED coordinates.
        """
        return self.state[3:6]

    @property
    def heading(self):
        """
        Returns the heading of the AUV wrt true north.
        """
        return geom.ssa(self.state[5])

    @property
    def pitch(self):
        """
        Returns the pitch of the AUV wrt NED.
        """
        return geom.ssa(self.state[4])

    @property
    def roll(self):
        """
        Returns the roll of the AUV wrt NED.
        """
        return geom.ssa(self.state[3])

    @property
    def relative_velocity(self):
        """
        Returns the surge, sway and heave velocity of the AUV.
        """
        return self.state[6:9]

    @property
    def relative_speed(self):
        """
        Returns the length of the velocity vector of the AUV.
        """
        return np.linalg.norm(self.relative_velocity)

    @property
    def angular_velocity(self):
        """
        Returns the rate of rotation about the NED frame.
        """
        return self.state[9:12]
    
    @property
    def chi(self):
        """
        Returns the rate of rotation about the NED frame.
        """
        [N_dot, E_dot, D_dot] = self.position_dot
        return np.arctan2(E_dot, N_dot)
        
    @property
    def upsilon(self):
        """
        Returns the rate of rotation about the NED frame.
        """
        [N_dot, E_dot, D_dot] = self.position_dot
        return np.arctan2(-D_dot, np.sqrt(N_dot**2 + E_dot**2))


def _surge(surge):
    surge = np.clip(surge, 0, 1)
    return surge*ss.thrust_max


def _steer(steer):
    steer = np.clip(steer, -1, 1)
    return steer*ss.rudder_max
