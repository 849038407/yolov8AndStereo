import numpy as np
import cv2
import time
from ultralytics import YOLO

from stereoCamera.stereoconfig import stereoCamera

config = stereoCamera()

# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
def getMiddleBurryParams(height, width, config):
    #读取内外参数
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

    return rectifyed_img1, rectifyed_img2


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output

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
    disparity  = stereo.compute(left_image, right_image)


    return disparity


if __name__ == '__main__':

    # 加载yolov8模型
    yolo = YOLO("./weight/best.pt")

    height = 480
    width = 1280

    camera = cv2.VideoCapture(1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = 0
    while True:
        # 开始计时
        t1 = time.time()
        ret, frame = camera.read()
        # 裁剪坐标
        # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
        left_frame = frame[0:480, 0:640]
        right_frame = frame[0:480, 640:1280]

        # 校正
        # 获取校正矩阵
        map1x, map1y, map2x, map2y, Q = getMiddleBurryParams(height, width/2, config)
        # 畸变校正和立体校正
        # 转化为灰度
        imgL = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        rectifimgL, rectifimgR = rectifyImage(imgL, imgR, map1x, map1y, map2x, map2y)
        # 转化为BGR
        imageL = cv2.cvtColor(rectifimgL, cv2.COLOR_GRAY2BGR)
        imageR = cv2.cvtColor(rectifimgR, cv2.COLOR_GRAY2BGR)
        output = draw_line(imageL, imageR)

        # 生成视差图
        # 计算视差
        disparity = stereoMatchSGBM(imageL, imageR).astype(np.float32)/16.0
        # 归一化函数算法，生成深度图（灰度图）
        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 生成深度图（颜色图）
        dis_color = disparity
        dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 伪彩色图
        dis_color = cv2.applyColorMap(dis_color, 2)

        # 计算三维坐标数据值
        threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=False)
        # 计算出的threeD，需要乘16，才等于现实中的距离
        # 有的文档是对disparity除以16

        # cv2.namedWindow("depth")
        # 鼠标回调事件
        # cv2.setMouseCallback("depth", onmouse_pick_points, threeD)
        '''
        # yolov8检测
        results = yolo(left_frame)
        for result in results:
            position = result.boxes.xywh
            for point in position:
                # 提取出每个目标框的中心像素坐标
                x_center, y_center, width, height = point.tolist()
                print("x_center=", x_center)
                print("y_center=", y_center)
                # 利用每个目标框的像素坐标进行坐标转化求出实际坐标值
                # x_3D, y_3D, z_3D= pixToWorld(x_center, y_center, disparity)

        annotated_frame = results[0].plot()
        '''

        # 完成计时,计算帧率
        fps = (fps + (1./(time.time() - t1))) / 2
        frame = cv2.putText(frame,  "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # res = stackImages(0.5, [[left_frame, right_frame], [dis_color, disp]])


        cv2.imshow("left", imageL)
        cv2.imshow("right", imageR)
        cv2.imshow("depth", dis_color)
        # cv2.imshow("res", res)
        cv2.imshow("WIN_NAME", disp)  # 显示深度图的双目画面
        cv2.imshow("Rectified Images with Lines", output)
        # cv2.imshow("yolov8", annotated_frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    camera.release()
    cv2.destroyWindow("left")
    cv2.destroyWindow("right")
    cv2.destroyWindow("depth")
    # cv2.destroyWindow("res")
    cv2.destroyWindow("WIN_NAME")
    cv2.destroyWindow("Rectified Images with Lines")
