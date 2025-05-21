import cv2  # 导入OpenCv库
import mediapipe as mp  # 导入Mediapipe库
import time

cap = cv2.VideoCapture(0)  # 0为打开默认摄像头,1为打开你设备列表的第二个摄像头,以此类推;
mpHands = mp.solutions.hands  # 使用Mediapipe库的手部姿势估计模型
hands = mpHands.Hands(static_image_mode=False, max_num_hands=4, model_complexity=1, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)  # 创建手部姿势估计器对象，设置参数。
mpDraw = mp.solutions.drawing_utils  # 初始化Mediapipe库绘图工具
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)  # 设置绘制手部关键点和连接线的样式
pTime = 0
cTime = 0  # 用于计算帧率

while True:  # 无限循环
    ret, img = cap.read()  # 读取摄像头的图像帧
    img = cv2.flip(img, 1)  # 对img图像进行水平翻转
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从BGR格式转换为RGB格式
        result = hands.process(imgRGB)  # 使用手部姿势估计器处理图像，获取结果
        # print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]  # 获取图像的高度，并将其赋值给变量imgHeight(其中[0]表示高度的维度)
        imgWidth = img.shape[1]  # 其中，img是一个图像对象，而shape[1]表示图像的宽度

        if result.multi_hand_landmarks:  # 检查是否检测到手部
            for handLms in result.multi_hand_landmarks:  # 遍历检测到的手部
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)  # 绘制手部关键点和连接线
                for i, lm in enumerate(handLms.landmark):  # 遍历每个关键点
                    xPos = int(lm.x * imgWidth)  # 计算关键点在图像中的x坐标
                    yPos = int(lm.y * imgHeight)  # 计算关键点在图像中的y坐标
                    # cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), )
                    if i == 4:  # 绘制特定关键点的标记,如果是特定的关键点（在代码中是第5个关键点）
                        cv2.circle(img, (xPos, yPos), 15, (92, 65, 214), cv2.FILLED)
                    print(i, xPos, yPos)  # 在特定关键点处绘制一个填充的圆

        cTime = time.time()  # 获取当前时间
        fps = 1 / (cTime - pTime)  # 计算帧率
        pTime = cTime  # 更新上一帧的时间
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)  # 在图像上显示帧率

        cv2.imshow('img', img)  # 显示处理后的图像

    if cv2.waitKey(1) == ord(' '):
        break  # 如果按下 ' ' 键，则退出循环



