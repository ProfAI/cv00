import cv2
from datetime import datetime

rec = False
bg_mode = False
dt_mode = False

cap = cv2.VideoCapture(0)
codec = cv2.VideoWriter_fourcc(*'MJPG')
out = None

ret, _ = cap.read()

if(not ret):
    print("Webcam non disponibile")
    exit(0)

while(cap.isOpened()):

    _, frame = cap.read()

    if(bg_mode):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(dt_mode):
        now = datetime.now()
        str_now = now.strftime("%d/%m/%Y %H:%M:%S")
        cv2.putText(frame, str_now, (20, frame.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
    if(rec):
        out.write(frame)
        cv2.circle(frame, (frame.shape[1]-30, frame.shape[0]-30), 10, (0, 0, 255), cv2.FILLED)

    cv2.imshow("webcam", frame)
    k = cv2.waitKey(1)
    if(k==ord("b")):
        bg_mode = not bg_mode
        print("Modalità bianco/nero: %s" % bg_mode)
    elif(k==ord("t")):
        dt_mode = not dt_mode
        print("Mostra data e ora: %s" % dt_mode)
    elif(k==ord("c")):
        now = datetime.now()
        filename = now.strftime("%Y%m%d%H%M%S")+".jpg"
        cv2.imwrite(filename, frame)
        print("Immagine catturata: %s" % filename)
    elif(k==ord(" ")):
        if(out==None):
            out = cv2.VideoWriter('output.avi', codec, 20., (640, 480))
        rec = not rec
        print("Registrazione: %s" % rec)
    elif(k==ord("q")):
        break

if(out!=None):
    out.release()

cap.release()
cv2.destroyAllWindows()