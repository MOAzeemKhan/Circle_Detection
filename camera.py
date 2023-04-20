import cv2 as cv
import numpy as np

prevCircle = None

dist = lambda x1,y1,x2,y2:(x1-x2)**2+(y1-y2)**2

class VideoCamera(object):
    def __init__(self):
        self.video = cv.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        blurFrame = cv.GaussianBlur(grayFrame, (17, 17), 0)

        circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, 100,
                                  param1=100, param2=30, minRadius=75, maxRadius=400)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            chosen = None
            for i in circles[0, :]:
                if chosen is None: chosen = i
                if prevCircle is not None:
                    if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0],
                                                                                        prevCircle[1]):
                        chosen = i
            cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
            cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255, 0, 255), 3)
            prevCircle = chosen
        ret, jpeg = cv.imencode('.jpg', frame)
        return jpeg.tobytes()