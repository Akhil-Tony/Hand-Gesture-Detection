import time
import cv2 as cv
import mediapipe as mp
from tensorflow import keras
import numpy as np

def preprocess(x,y):
    joined =  x+y
    joined_arr = np.array(joined)
    return joined_arr

def findLabel(prediction):
    dominant_pred = np.argmax(prediction)
    label = class_label[dominant_pred]
    prob = prediction[0][dominant_pred]
    prob = np.round(prob,2)
    return label,prob

class_label = ['Hai','Well Done','Victory','Fuck Off']

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)

mpDraw = mp.solutions.drawing_utils
ctime = 0
ptime = 0

cam0 = cv.VideoCapture(1)

gesture_model = keras.models.load_model('gesture_model.h5')

while True:
    success,frame = cam0.read()
    frame = cv.flip(frame,1)
    h,w,c = frame.shape
    frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if success:
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                xs = []
                ys = []
                mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)
                for idx,lm in enumerate(handLms.landmark):
                    xs.append(lm.x)
                    ys.append(lm.y)
                processed_instance = preprocess(xs,ys)
                prediction = gesture_model.predict(processed_instance.reshape(1,42))
                label,prob = findLabel(prediction)
                cv.putText(frame,label,(10,200),cv.FONT_HERSHEY_COMPLEX,2,(0,233,0),thickness=2)
#                 cv.putText(frame,str(prob)+' %',(10,220),cv.FONT_HERSHEY_COMPLEX,1,(200,0,0),thickness=2)
        ctime = time.time()
        fps = int(1/(ctime-ptime))
        ptime = ctime
        cv.putText(frame,str(fps),(10,70),cv.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
cam0.release()
cv.destroyAllWindows()
