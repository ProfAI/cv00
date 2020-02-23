import cv2

img = cv2.imread("res/cat.jpg")
integral = cv2.integral(img)

print(img[:3,:3,0])
print(integral[:4,:4,0])