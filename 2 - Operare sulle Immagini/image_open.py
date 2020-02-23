import cv2

img_path = "res/elon.jpg"

# Apre l'immagine a colori
img = cv2.imread('res/elon.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)

# Apre l'immagine in bianco e nero
img = cv2.imread('res/elon.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)

cv2.destroyAllWindows()