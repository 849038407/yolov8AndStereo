import cv2
import numpy as np
import time
from ultralytics import YOLO

from stereoCamera.stereoconfig import stereoCamera

config = stereoCamera()


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
def getMiddleBurryParams(height, width, config):
    # 读取内外参数
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变化
    height = int(height)
    width = int(width)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    rectifyed_img1 = rectifyed_img1[0:480, 0:640]
    rectifyed_img2 = rectifyed_img2[0:480, 0:640]

    return rectifyed_img1, rectifyed_img2


# 视差计算
def stereoMatchSGBM(left_image, right_image):
    # SGBM匹配参数设置
    ###########################################################################
    # ------------------------------------SGBM算法----------------------------------------------------------
    #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    #                               取16、32、48、64等
    #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    # ------------------------------------------------------------------------------------------------------
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3;
    paraml = {
        'minDisparity': 0,
        'numDisparities': 128,
        'blockSize': blockSize,
        'P1': 8 * img_channels * blockSize ** 2,
        'P2': 32 * img_channels * blockSize ** 2,
        'disp12MaxDiff': 1,
        'preFilterCap': 63,
        'uniquenessRatio': 15,
        'speckleWindowSize': 100,
        'speckleRange': 1,
        'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
    }

    # 构建SGBM对象
    stereo = cv2.StereoSGBM_create(**paraml)

    # 计算视差图
    disparity = stereo.compute(left_image, right_image)
    # print("disparity.shape=", disparity.shape)
    # print("disparity.shape[0]=", disparity.shape[0])
    # print("disparity.shape[1]=", disparity.shape[1])
    pad_width = ((480 - disparity.shape[0]), (640 - disparity.shape[1]))
    disparity = np.pad(disparity, pad_width, mode='constant', constant_values=-1)
    return disparity


if __name__ == '__main__':

    height = 480
    width = 1280

    camera = cv2.VideoCapture(1)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    fps = camera.get(cv2.CAP_PROP_FRAME_COUNT)

    yolo = YOLO("./weight/yolov8n.pt")
    while camera.isOpened():
        # 开始计时
        t1 = time.time()
        ret, frame = camera.read()
        # 裁剪坐标
        # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
        left_frame = frame[0:480, 0:640]
        right_frame = frame[0:480, 640:1280]

        # 校正
        # 获取校正矩阵
        map1x, map1y, map2x, map2y, Q = getMiddleBurryParams(height, width / 2, config)
        # 畸变校正和立体校正
        # 转化为灰度
        imgL = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        rectifyImgL, rectifyImgR = rectifyImage(imgL, imgR, map1x, map1y, map2x, map2y)
        # 转化为BGR
        imageL = cv2.cvtColor(rectifyImgL, cv2.COLOR_GRAY2BGR)
        imageR = cv2.cvtColor(rectifyImgR, cv2.COLOR_GRAY2BGR)
        print("rectifyImgL.shape=", rectifyImgL.shape)
        print("rectifyImgR.shape=", rectifyImgR.shape)

        # 生成视差图
        # 计算视差
        disparity = stereoMatchSGBM(rectifyImgL, rectifyImgR).astype(np.float32) / 16.0

        # 归一化
        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 生成深度图（颜色图）
        dis_color = disparity
        dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 伪彩色图
        dis_color = cv2.applyColorMap(dis_color, 2)

        # 计算三维坐标数据值
        threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)

        # 问题是每一次校正后图片的像素大小会发生改变
        # 通过threeD来获取3D坐标

        results = yolo(right_frame)
        for result in results:
            for i in range(result.boxes.shape[0]):
                position = result.boxes.xywh[i]
                # 提取出每个目标框的中心像素坐标
                x_center, y_center, width, height = position.tolist()
                x_center = int(x_center)
                y_center = int(y_center)
                print("x_center=", x_center)
                print("y_center=", y_center)
                # 利用每个目标框的像素坐标进行坐标转化求出实际坐标值
                world_x, world_y, world_z = threeD[y_center][x_center][0] / 1000.0, threeD[y_center][x_center][
                    1] / 1000.0, threeD[y_center][x_center][2] / 1000.0
                print("world_x=", world_x)
                print("world_y=", world_y)
                print("world_z=", world_z)
                # 将三维坐标转化成字符串
                coords_text = f"X: {world_x:.2f}, Y: {world_y:.2f}, Z: {world_z:.2f}"
                # 在图像上显示坐标文本
                cv2.putText(right_frame, coords_text, (x_center, y_center - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
        annotated_frame = results[0].plot()

        # 完成计时,计算帧率
        fps = (fps + (1. / (time.time() - t1))) / 2
        frame = cv2.putText(annotated_frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("yolov8", annotated_frame)

        # 释放不再需要的变量
        disparity = None
        disp_color = None
        threeD = None

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    camera.release()
    cv2.destroyWindow("yolov8")
