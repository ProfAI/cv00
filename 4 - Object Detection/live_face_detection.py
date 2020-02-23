import cv2

def contains(r1, r2):
     return r1[0] < r2[0] < r2[0]+r2[2] < r1[0]+r1[2] and r1[1] < r2[1] < r2[1]+r2[3] < r1[1]+r1[3]

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

if(not cap.read()[0]):
    print("Webcam non è disponibile")
    exit(0)

while(cap.isOpened()):

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 15)
    smiles = smile_cascade.detectMultiScale(gray, 1.4, 45)

    for f in faces:
        cv2.rectangle(frame, (f[0], f[1]), (f[0]+f[2], f[1]+f[3]), (0, 255, 0), 2)
        for s in smiles:
            if(contains(f,s)):
                cv2.rectangle(frame, (s[0], s[1]), (s[0]+s[2], s[1]+s[3]), (255, 0, 0), 2)
                img_face = frame[f[1]:f[1]+f[3], f[0]:f[0]+f[2]]
                cv2.imwrite("res/shot.jpg", img_face)

    cv2.imshow("Webcam", frame)

    if(cv2.waitKey(1)==ord("q")):
        break

cap.release()
cv2.destroyAllWindows()