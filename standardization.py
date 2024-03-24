'''
相机参数获取通过matlab
此代码不准确
'''

import cv2
import numpy as np
import os

def getImageList(img_dir):
    # 获取图片文件夹的位置，方便opencv读取
    # 参数：照片路径
    # 返回值：数组，每一个元素表示一张照片的绝对路径
    imgPath = []
    if os.path.exists(img_dir) is False:
        print("error dir")
    else:
        for parent, dirNames, fileNames in os.walk(img_dir):
            for fileName in fileNames:
                imgPath.append(os.path.join(parent, fileName))
    return imgPath

def getObjectPoubts(m, n, k):
    # 计算真实坐标
    # 参数:内点行数,内点列,标定板大小
    # 返回值:数组,(m*n行,3列) 相当于m*n个点(x,y,z)坐标
    objP = np.zeros(shape=(m*n, 3), dtype=np.float32)
    for i in range(m*n):
        objP[i][0] = i % m
        objP[i][1] = int(i/m)

    return objP*k

if __name__ == '__main__':
    # 相机标定参数设定（单目，双目）
    # 定义迭代停止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 计算标定板真是坐标,假设板内点（）, 大小10mm*10mm
    objPoint = getObjectPoubts(7, 11, 10)

    objPoints = []
    imgPointsL = []
    imgPointsR = []


    # 相机路径
    imgPathL = "F:/Yolov8_CarDetect/stereoCamera/left"
    imgPathR = "F:/Yolov8_CarDetect/stereoCamera/right"
    filePathL = getImageList(imgPathL)
    filePathR = getImageList(imgPathR)
    length = len(filePathL)

    # 检测是否读取到照片
    # print(filePathL)
    # print(length)

    # 测试是否转变成灰色图
    """
    test_img = cv2.imread(filePathL[2])
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayL", test_gray)
    key = cv2.waitKey(0)  # 等待按键命令, 1000ms 后自动关闭
    """
    for i in range(length):
        # 分别读取每张图片并转化为灰度图
        imgL = cv2.imread(filePathL[i])
        imgR = cv2.imread(filePathR[i])

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)


        # print(grayL)
        # opencv寻找角点
        retL, cornersL = cv2.findChessboardCorners(grayL, (7, 11), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (7, 11), None)
        # 目前问题 retL与retR都为False ----已解决

        if (retL & retR) is True:
            # opencv对真实坐标格式要求, vector<vector<Point3f>>类型
            objPoints.append(objPoint)
            # 角点细化
            cornersL2 = cv2.cornerSubPix(grayL, cornersL, (5, 5), (-1, -1), criteria)
            cornersR2 = cv2.cornerSubPix(grayR, cornersR, (5, 5), (-1, -1), criteria)
            imgPointsL.append(cornersL2)
            imgPointsR.append(cornersR2)

    # print(imgPointsL)
    # print(imgPointsR)

    # 对左右相机分别进行单目相机标定
    retL, cameraMatrixL, distMatrixL, RL, TL = cv2.calibrateCamera(objPoints, imgPointsL, (640, 480), None, None)
    retR, cameraMatrixR, distMatrixR, RR, TR = cv2.calibrateCamera(objPoints, imgPointsR, (640, 480), None, None)

    # 双目相机校正
    retS, mLS, dLS, mRS, dRS, R, T, E, F = cv2.stereoCalibrate(objPoints, imgPointsL,
                                                               imgPointsR, cameraMatrixL,
                                                               distMatrixL, cameraMatrixR,
                                                               distMatrixR, (640, 480),
                                                               criteria_stereo, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    # 标定结束，结果输出，cameraMatrixL，cameraMatrixR分别为左右相机内参数矩阵
    # R， T为相机2与相机1旋转平移矩阵

    print(cameraMatrixL)
    print('*' * 20)
    print(cameraMatrixR)
    print('*' * 20)
    print(R)
    print('*' * 20)
    print(T)