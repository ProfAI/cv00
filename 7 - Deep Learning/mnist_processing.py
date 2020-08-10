import cv2
import os

DATASET_DIR = "../datasets/mnist/"
IMAGES_DIR = DATASET_DIR+"images/"
DATASET_FILENAME = "mnist.csv"

data_file = open(DATASET_DIR+DATASET_FILENAME,"w")

print("Lettura di tutte le immagini da %s" % IMAGES_DIR)
print("Scrittura sul file %s" % DATASET_FILENAME)

counter = [0]*10
total = 0

for i in range(10):
    current_dir = IMAGES_DIR+str(i)
    for f in os.listdir(current_dir):

        if(not ".jpg" in f):
            continue

        img = cv2.imread(current_dir+"/"+f, cv2.IMREAD_GRAYSCALE)
        arr = img.flatten()
        data_file.write(",".join(arr.astype(str)))
        data_file.write(","+str(i))
        data_file.write("\n")

        counter[i]+=1
        total+=1

data_file.close()

print("Totale immagini scritte: %d" % total)
print("Immagini scritte per classe: %s" % counter)