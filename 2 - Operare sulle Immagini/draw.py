import cv2

TICKNESS = 2
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLU = (255, 0, 0)
BLACK = (0,0,0)

# Apriamo l'immagine
img = cv2.imread('res/elon.jpg')
cv2.imshow('image', img)
cv2.waitKey(0)

# Disegniamo un quadrato
img_h, img_w = img.shape[0], img.shape[1]
l = 200
center = (img_w//2, img_h//2)
cv2.rectangle(img, (center[0]-l//2, center[1]-l//2), (center[0]+l//2, center[1]+l//2), RED, TICKNESS)
cv2.imshow("image",img)
cv2.waitKey(0)

# Disegniamo un cerchio
r = 10
cv2.circle(img, center, r, GREEN, TICKNESS)
cv2.imshow("image",img)
cv2.waitKey(0)

# Disegniamo due linee
cv2.line(img, (center[0],0), (center[0], img_h), BLU, TICKNESS)
cv2.imshow("image",img)
cv2.waitKey(0)

cv2.line(img, (0, center[1]), (img_w, center[1]), BLU, TICKNESS)
cv2.imshow("image",img)
cv2.waitKey(0)

# Scriviamo del testo
cv2.putText(img, "@Giuseppe Gullo", (img_w, img_h-20), cv2.FONT_HERSHEY_PLAIN, 2, BLACK, TICKNESS)
cv2.imshow("image",img)
cv2.waitKey(0)
