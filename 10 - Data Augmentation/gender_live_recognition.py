import cv2
from tensorflow.keras.models import load_model

SCALE = (200, 200)

model = load_model('model_augmented.h5')

cap = cv2.VideoCapture(0)

if(not cap.read()[0]):
    print("Webcam non è disponible")
    exit(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(cap.isOpened()):

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = face_cascade.detectMultiScale(gray, 1.1, 15)

    for rect in rects:

        img = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

        small_img = cv2.resize(img, SCALE)
        x = small_img.astype(float)
        x/=255.

        x = x.reshape(1, SCALE[0], SCALE[1], 3)
        y = model.predict(x)

        y = y[0][0]

        label = "Uomo" if y>0.5 else "Donna"
        percentage = y if y>0.5 else 1.0-y
        percentage = round(percentage*100,1)

        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 2)
        cv2.rectangle(frame, (rect[0], rect[1]-20), (rect[0]+170, rect[1]), (0,255,0), cv2.FILLED)
        cv2.putText(frame, label+" ("+str(percentage)+"%)", (rect[0]+5, rect[1]), cv2.FONT_HERSHEY_PLAIN, 1.4, (255,255,255), 2)

    cv2.imshow("Gender Recognition", frame)

    if(cv2.waitKey(1)==ord("q")):
        break