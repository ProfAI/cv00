import cv2
from tensorflow.keras.models import load_model
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Gender classification from picture.')
parser.add_argument('img', help='insert the image path')
args = parser.parse_args()

SCALE = (200, 200)

model = load_model('model_augmented.h5')

img = cv2.imread(args.img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
rects = face_cascade.detectMultiScale(gray, 1.1, 15)

for rect in rects:

    face = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    face = cv2.resize(face, SCALE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    x = face.astype(float)
    x/=255.

    x = x.reshape(1, SCALE[0], SCALE[1], 3)
    y = model.predict(x)

    y = y[0][0]
    print(y)

    label = "Uomo" if y>0.5 else "Donna"
    percentage = y if y>0.5 else 1.0-y
    percentage = round(percentage*100,1)

    cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 2)
    cv2.rectangle(img, (rect[0], rect[1]-20), (rect[0]+170, rect[1]), (0,255,0), cv2.FILLED)
    cv2.putText(img, label+" ("+str(percentage)+"%)", (rect[0]+5, rect[1]), cv2.FONT_HERSHEY_PLAIN, 1.4, (255,255,255), 2)

cv2.imshow("Gender Recognition", img)

if cv2.waitKey(0)==ord("q"):
    cv2.destroyAllWindows()
