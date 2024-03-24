import numpy as np

# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[550.27, 0, 303.75],
                                         [0, 548.63, 251.54],
                                         [0, 0, 1]])
        # 右相机内参
        self.cam_matrix_right = np.array([[548.69, 0, 325.83],
                                          [0, 547.06, 253.85],
                                          [0, 0, 1]])

        # 左右相机畸变系数[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.0831, 0.3541, -0.000241651464801373, -0.000816896189186328, -0.4986]])
        self.distortion_r = np.array([[-0.0907, 0.4267, 0.000469374116031005, 0, -0.6204]])


        # 根据实验需要改变重新计算
        # 旋转矩阵
        self.R = np.array([[1, 0, 0.02],
                           [0, 1, -0.01],
                           [-0.02, 0.01, 1]])

        # 平移矩阵
        self.T = np.array([[-55.85],
                           [0.16],
                           [-0.97]])

        # 主点列坐标的差
        self.doffs = 0.0

        # 指示上述内外参是否经过立体校正得到的结果
        self.isRectified = False







