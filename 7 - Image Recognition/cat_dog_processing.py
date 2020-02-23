import cv2
import os

DATASET_DIR = "../datasets/cat_dog/"
IMAGES_DIR = DATASET_DIR+"images/"
SCALE = (64, 64)

out = open(DATASET_DIR+"cat_dog.csv","w")
count = 0

for f in os.listdir(IMAGES_DIR):
    if(".jpg" in f):
        if("cat" in f):
            label="1"
        elif("dog" inW f):
            label="0"
        else:
            continue
        
        img = cv2.imread(IMAGES_DIR+f, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, SCALE)
        img = img.flatten().astype(str)
        data = str(count)+","+",".join(img)+","+label
        out.write(data+"\n")
        count+=1
        

out.seek(out.tell()-2, os.SEEK_SET)
out.truncate()
out.close()

print("%d immagini scritte" % count)
