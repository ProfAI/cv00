import cv2

img_path = input("Inserisci il percorso all'immagine: ")
img = cv2.imread(img_path)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 20)

print("Volti trovati: %d" % len(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imwrite("italia_faces.jpg",img)