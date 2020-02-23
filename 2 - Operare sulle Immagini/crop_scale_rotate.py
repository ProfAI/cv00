import cv2

img_path = "res/elon.jpg"

# Apre l'immagine
img = cv2.imread('res/elon.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)

# Ridimensiona l'immagine
img_w, img_h = 400, 550
img_resized = cv2.resize(img, (img_w, img_h))
cv2.imshow('image',img_resized)
cv2.waitKey(0)

# Ritagliamo l'immagine
size = 200
img_cropped = img_resized[img_h//2-size//2:img_h//2+size//2, img_w//2-size//2:img_w//2+size//2]
#cv2.line(img_resized, (img_w//2-size//2, img_h//2-size//2), (img_w//2+size//2, img_h//2+size//2), (0,255,0), 2)
cv2.imshow('image',img_cropped)
cv2.waitKey(0)

# Ruotiamo l'immagine
angle =  180
center = (img_w/2, img_h/2)
rot_mat = cv2.getRotationMatrix2D(center,angle,1.)
img_rotated = cv2.warpAffine(img_resized, rot_mat, (img_w, img_h))
print(img_rotated.shape)
cv2.imshow('image',img_rotated)
cv2.waitKey(0)

# Salviamo le immagini
cv2.imwrite("res/elon_resized.jpg", img_resized)
cv2.imwrite("res/elon_cropped.jpg", img_cropped)
cv2.imwrite("res/elon_rotated.jpg", img_rotated)

cv2.destroyAllWindows()