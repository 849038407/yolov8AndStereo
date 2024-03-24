import cv2
from ultralytics import YOLO

yolo = YOLO("./weight/yolov8n.pt")

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = yolo(frame)
        for result in results:
            position = result.boxes.xywh
            for point in position:
                x_center, y_center, width, height = point.tolist()
                # print("中心坐标为{},{}".format(x_center, y_center)
        annotated_frame = results[0].plot()
        cv2.imshow(winname="YOLOV8", mat=annotated_frame)
        # 按ESC退出
        if cv2.waitKey(1) == 27:
            break
# 释放连接
cap.release()
cv2.destroyAllWindows()
