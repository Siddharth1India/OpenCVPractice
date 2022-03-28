import cv2
print("Package Imported")

# '''Image
# '''

# # img = cv2.imread("./static/flexbox.png")
# # cv2.imshow("Output",img)
# # cv2.waitKey(10000)


# '''Video
# '''
vid = cv2.VideoCapture(0)
vid.set(3, 640)
vid.set(4, 480)
vid.set(10, 1000)

while True:
    success, img = vid.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

# '''Functions
# '''
# '''Gray Scale
# '''
# # img = cv2.imread("./static/flexbox.png")

# # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # cv2.imshow("Output", imgGray)
# # cv2.waitKey(0)s

# '''Blur
# '''
# # img = cv2.imread("./static/flexbox.png")

# # imgBlur = cv2.GaussianBlur(img, (29,29), 0)
# # cv2.imshow("Output", imgBlur)
# # cv2.waitKey(0)

# '''Edge Detector
# '''
# # img = cv2.imread("./static/flexbox.png")
# # imgCanny = cv2.Canny(img, 10, 10)
# # cv2.imshow("Canny",imgCanny)
# # cv2.waitKey(0)

# '''Dialation
# '''
# import numpy as np

# # kernel = np.ones((5,5), np.uint8)
# # # # print(kernel)
# # img = cv2.imread("./static/flexbox.png")
# # imgCanny = cv2.Canny(img, 10, 10)
# # imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
# # cv2.imshow("ImgDialation", imgDialation)
# # cv2.waitKey(0)

# '''Eroded
# '''
# # imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
# # imgEroded = cv2.erode(imgDialation, kernel, iterations=1)
# # cv2.imshow("ImgEroded", imgEroded)
# # cv2.waitKey(0)

# '''OpenCV Resize and Crop
# '''
# # img = cv2.imread("./static/flexbox.png")
# # # print(img.shape)
# # imgResize = cv2.resize(img, (640, 480))

# '''In normal OpenCV, We have Width x Height
# But here in matrix operation to crop, height is first
# Overall, Matrix Operation: Height x Width
# '''
# # imgCrop = imgResize[0:280,0:520]
# # cv2.imshow("Output", imgCrop)
# # cv2.waitKey(0)

# '''Shapes and Text
# '''
# # img = np.zeros((512,512, 3), np.uint8)
# # # img[200:300, 200:300] = 255,0,0
# # # print(type(img.shape[0]))
# # # # cv2.line(img, (0,0), (512,512), (0,130,255), 3)
# # # cv2.rectangle(img, (0,0), (250, 350), (0, 135, 255), cv2.FILLED)
# # # cv2.circle(img, (200, 200), 100, (0, 135, 255), 8)
# # cv2.putText(img, "OpenCV", (20, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 135, 255), 3)
# # cv2.imshow("image" ,img)
# # cv2.waitKey(0)

# '''Warp Perspective
# '''
# # img = cv2.imread("./static/cards.png")

# # width = 250
# # height = 350
# # pts1 = np.float32([[64, 127],[164, 110],[92, 277],[202, 251]])
# # pts2 = np.float32([[0,0],[width,0],[0, height],[width, height]])

# # matrix = cv2.getPerspectiveTransform(pts1, pts2)

# # imgOutput = cv2.warpPerspective(img, matrix, (250, 350))
# # cv2.imshow("Output", imgOutput)
# # cv2.waitKey(0)

# '''Joining Images
# '''
# # img = cv2.imread("./static/flexbox.png")
# # # img = img[200:300, 200:300]
# # # hor = np.hstack((img, img))
# # # ver = np.vstack((img, img))
# # # cv2.imshow("Output", ver)


# # cv2.waitKey(0)

# '''Color Detection
# '''

# # def empty(a):
# #     pass
# # cv2.namedWindow("TrackBar")
# # cv2.resizeWindow("TrackBar", 640, 240)
# # cv2.createTrackbar("HueMin", "TrackBar", 0, 179, empty)
# # cv2.createTrackbar("HueMax", "TrackBar", 20, 179, empty)
# # cv2.createTrackbar("SatMin", "TrackBar", 65, 255, empty)
# # cv2.createTrackbar("SatMax", "TrackBar", 240, 255, empty)
# # cv2.createTrackbar("ValMin", "TrackBar", 116, 255, empty)
# # cv2.createTrackbar("ValMax", "TrackBar", 255, 255, empty)
# # path = "./static/car.png"
# # while True:
# #     img = cv2.imread(path)
# #     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# #     h_min = cv2.getTrackbarPos("HueMin", "TrackBar")
# #     h_max = cv2.getTrackbarPos("HueMax", "TrackBar")
# #     s_min = cv2.getTrackbarPos("SatMin", "TrackBar")
# #     s_max = cv2.getTrackbarPos("SatMax", "TrackBar")
# #     v_min = cv2.getTrackbarPos("ValMin", "TrackBar")
# #     v_max = cv2.getTrackbarPos("ValMax", "TrackBar")
# #     lower = np.array([h_min, s_min, v_min])
# #     upper = np.array([h_max, s_max, v_max])
# #     mask = cv2.inRange(imgHSV, lower, upper)
# #     imgResult = cv2.bitwise_and(img,img, mask=mask)
# #     # cv2.imshow("Output", imgHSV)
# #     # cv2.imshow("Mask", mask)
# #     cv2.imshow("Result", imgResult)
# #     cv2.waitKey(1)

# '''Contour/ Shape Detector
# '''

# # def getContours(img):
# #     contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# #     for cnt in contours:
# #         area = cv2.contourArea(cnt)
# #         print(area)
# #         if area>500:
# #             cv2.drawContours(imgBlank ,cnt, -1, (0,135,255), 2)
# #             peri = cv2.arcLength(cnt, True)
# #             approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
# #             x, y, w, h = cv2.boundingRect(approx)
# #             objCor = len(approx)
# #             ObjType = ""
# #             if objCor==3:   ObjType="Triangle"
# #             elif objCor==4:
# #                 aspectRatio = w/float(h)
# #                 if aspectRatio >0.95 and aspectRatio<1.05:
# #                     ObjType = "Square"
# #                 else:
# #                     ObjType = "Rectangle"
# #             elif objCor>4:  ObjType = "Circle"
# #             else: ObjType="None"
# #             cv2.rectangle(imgBlank, (x, y), (x+w, y+h), (0,255,0), 1)
# #             cv2.putText(imgBlank, ObjType, ((x+(w//2)-10), y+(h//2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),1)
# # img = cv2.imread("./static/shape.png")
# # imgBlank = img.copy()

# # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # imgBlur = cv2.GaussianBlur(imgGray, (3,3), 1)
# # imgCanny2 = cv2.Canny(imgBlur, 50, 50)

# # cv2.imshow("Output", img)
# # cv2.imshow("Gray", imgGray)
# # cv2.imshow("Blur", imgBlur)
# # # cv2.imshow("Canny2", imgCanny)
# # cv2.waitKey(0)

# # getContours(imgCanny2)
# # cv2.imshow("imgBlank", imgBlank)
# # cv2.waitKey(0)

# '''Face Detection
# '''
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# img = cv2.imread("./static/face.png")
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = face_cascade.detectMultiScale(imgGray, 1.1, 4)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# cv2.imshow("Result", img)
# cv2.waitKey(0)