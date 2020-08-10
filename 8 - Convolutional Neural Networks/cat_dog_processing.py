import cv2
import os

DATASET_DIR = "../datasets/cat_dog_small/"
OUTFILE_NAME = "cat_dog.csv"
SCALE = (64, 64)

out = open(DATASET_DIR+OUTFILE_NAME, "w")

classes = {"cat":"1", "dog":"0"}
counter = {"cat":0, "dog":0}

print("Letture di tutte le immagini da %s" % DATASET_DIR)
print("Scrittura in %s" % OUTFILE_NAME)

for c in classes:
    
    current_dir = DATASET_DIR+c

    for f in os.listdir(current_dir):

        if(not ".jpg" in f):
            continue

        img = cv2.imread(current_dir+"/"+f, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, SCALE)
        img = img.flatten().astype(str)
        data = ",".join(img)+","+classes[c]
        out.write(data+"\n")
        
        counter[c]+=1

out.close()
print("Immagini scritte: %s" %counter)
