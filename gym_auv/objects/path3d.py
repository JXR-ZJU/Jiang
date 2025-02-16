import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import gym_auv.utils.geomutils as geom

from scipy.optimize import fminbound
from mpl_toolkits.mplot3d import Axes3D


class Path3D():
    def __init__(self, waypoints):
        self.waypoints = np.array(waypoints)
        self.nwaypoints = len(waypoints)
        self.segment_lengths = self._get_path_lengths() # wp之间的距离
        self.length = np.sum(self.segment_lengths) # 总长度
        self.azimuth_angles = np.array([]) # 每个wp处的方位角
        self.elevation_angles = np.array([]) # 每个wp处的俯仰角

        self._get_parametric_params()


    def __call__(self, s):
        seg_start, seg_index = self._get_segment_start(s)
        alpha = self.azimuth_angles[seg_index]
        beta = self.elevation_angles[seg_index]
        seg_distance = s - seg_start
        x_start, y_start, z_start = self.waypoints[seg_index]

        x = x_start + seg_distance*np.cos(alpha)*np.cos(beta)
        y = y_start + seg_distance*np.sin(alpha)*np.cos(beta)
        z = z_start - seg_distance*np.sin(beta)

        return np.array([x,y,z]) # 当前位置


    def _get_segment_start(self, s): # 当前在哪段（哪两个wp之间）
        seg_start = 0
        for i, sl in enumerate(self.segment_lengths):
            if s <= seg_start+sl:
                return seg_start, i
            else:
                seg_start += sl
        return seg_start, i


    def _get_parametric_params(self):
        diff = np.diff(self.waypoints, axis=0)
        for i in range(self.nwaypoints-1):
            derivative = diff[i] / self.segment_lengths[i] # 坐标变化率
            alpha = np.arctan2(derivative[1], derivative[0]) # 水平面夹角
            beta = np.arctan2(-derivative[2], np.sqrt(derivative[0]**2 + derivative[1]**2)) # 竖直面夹角
            self.azimuth_angles = np.append(self.azimuth_angles, geom.ssa(alpha)) # 不考虑axis，都将先展平成一维数组后沿着axis=0的方向添加
            self.elevation_angles = np.append(self.elevation_angles, geom.ssa(beta))


    def plot_path(self):
        x = []
        y = []
        z = []
        s = np.linspace(0, self.length, 10000) # 总长度切片
        
        for ds in s:
            x.append(self(ds)[0])
            y.append(self(ds)[1])
            z.append(self(ds)[2])

        ax = plt.axes(projection='3d')
        ax.plot(x, y, z)
        return ax
    
    
    def get_closest_s(self, position):
        s = fminbound(lambda s: np.linalg.norm(self(s) - position), # 在给定范围内找到函数的最小值
                    x1=0, x2=self.length, xtol=1e-6,
                    maxfun=10000) # 要最小化的目标函数，优化界限，收敛容差，允许的最大函数评估次数
        return s


    def get_closest_point(self, position): # 离position最近的点
        s = self.get_closest_s(position)
        return self(s)


    def _get_path_lengths(self):
        diff = np.diff(self.waypoints, axis=0) # 沿着指定轴计算第N维的离散差值
        seg_lengths = np.sqrt(np.sum(diff**2, axis=1))
        return seg_lengths

    
    def get_endpoint(self):
        return self(self.length)

    
    def get_direction_angles(self, s): # 当前到达的wp的方位角和俯仰角
        _, seg_index = self._get_segment_start(s)
        return self.azimuth_angles[seg_index], self.elevation_angles[seg_index]


def generate_random_waypoints(nwaypoints): # 随机生成nwaypoints个wp，
    waypoints = [np.array([0,0,0])]
    for i in range(nwaypoints-1):
        azimuth = np.random.normal(0,0.5) * np.pi/6 # 方位角和俯仰角都在15°以内
        elevation = np.random.normal(0,0.5) * np.pi/6
        dist = np.random.randint(50, 100)

        x = waypoints[i][0] + dist*np.cos(azimuth)*np.cos(elevation)
        y = waypoints[i][1] + dist*np.sin(azimuth)*np.cos(elevation)
        z = waypoints[i][2] - dist*np.sin(elevation)
        wp = np.array([x, y, z])
        waypoints.append(wp)
    return waypoints
