import cv2
import mediapipe as mp
import time
from google.protobuf.json_format import MessageToDict
from PIL import Image
import glob

class handPoseDetector():
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con= 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands,
                                        self.detection_con, self.track_con)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, hand_lms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, hand="Right", draw=True):
        lm_list = []

        if(self.results.multi_handedness):
            for idx, hand_handedness in enumerate(self.results.multi_handedness):
                hand_dict = MessageToDict(hand_handedness)
                print(hand_dict)
                hand_label = hand_dict["classification"][0]["label"]
                if (hand_label == hand):
                    i = hand_dict["classification"][0]["index"]
                    print("i: ", i, " size: ", len(self.results.multi_hand_landmarks))
                    if len(self.results.multi_hand_landmarks) == 1:
                        i = 0
                    else:
                        i = 1
                    myHand = self.results.multi_hand_landmarks[i]
                    for id, lm in enumerate(myHand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        #print(id,cx,cy)
                        lm_list.append([id,cx,cy])
                        if id == 4:
                            if draw:
                                cv2.circle(img, (cx, cy), 10, (255,0,255), cv2.FILLED)
        return lm_list
    """     
        if (self.results.multi_hand_landmarks):
            for i in range(len(self.results.multi_handedness)):
                print("Hand: ",i)
                print(self.results.multi_handedness[i])
    """


if __name__ == "__main__":

    handDetector = handPoseDetector(detection_con=0.01, max_hands=1, track_con=0.01)


    image_list = []
    for filename in glob.glob('/home/elena/Videos/hands/*png'):
        print(filename)
        imu = cv2.imread(filename)
        img = handDetector.find_hands(imu)
        cv2.imshow("Image", img)
        cv2.waitKey(0)

"""
    filename = "/home/elena/Videos/Yale-Grasp-Dataset/040.mp4"
    #filename = "/home/elena/Videos/Videos-GoPro/GH012471.MP4"

    cap = cv2.VideoCapture(filename)
    #num_frames = 400
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert (cap.isOpened())

    for index in range(560,num_frames):
        # read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            print(index)
            img = handDetector.find_hands(frame)
            lmList = handDetector.findPosition(img)
            if len(lmList) != 0:
                i = 1
                #print(lmList)
            cv2.imshow("Image", img)
            cv2.waitKey(1)

        num_frames += 1
"""
