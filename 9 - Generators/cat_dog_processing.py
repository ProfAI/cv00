import os
from shutil import copyfile

DATASET_DIR = "../datasets/cat_dog/"
IMAGES_DIR = DATASET_DIR+"images/"

if(not os.path.isdir(DATASET_DIR+"classes")):
    os.mkdir(DATASET_DIR+"classes")
    os.mkdir(DATASET_DIR+"classes/cats")
    os.mkdir(DATASET_DIR+"classes/dogs/")
    
for f in os.listdir(IMAGES_DIR):
    if(".jpg" in f):
        if("cat" in f):
            copyfile(IMAGES_DIR+f, DATASET_DIR+"classes/cats/"+f)
        elif("dog" in f):
            copyfile(IMAGES_DIR+f, DATASET_DIR+"classes/dogs/"+f)