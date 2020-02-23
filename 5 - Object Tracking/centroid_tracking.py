import cv2
import numpy as np
from scipy.spatial import distance as dist

def get_centroid(rect):
    return ((2*rect[0]+rect[2])//2, (2*rect[1]+rect[3])//2)

objects = {}
nextObjectID = 0

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not cap.read()[0]:
    print("Camera non disponibile")
    exit(0)

while(cap.isOpened()):

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 20)

    if(len(faces)!=0):

        foundCentroids = np.zeros((len(faces), 2), dtype="int")

        for i,f in enumerate(faces):
            foundCentroids[i] = get_centroid(f)
            
        if(len(objects)==0):
            for i in range(len(foundCentroids)):
                objects[nextObjectID] = foundCentroids[i]
                nextObjectID+=1
        else:
            objectIDs = list(objects.keys())
            objectCentroids = list(objects.values())
            
            D = dist.cdist(np.array(objectCentroids), foundCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                
                if row in usedRows or col in usedCols:
                    continue
                
                objectID = objectIDs[row]
                objects[objectID] = foundCentroids[col]
                usedRows.add(row)
                usedCols.add(col)
    
            newObjs = set(range(D.shape[1]))-set(usedCols)

            if(len(newObjs)!=0):
                for obj in newObjs:
                    objects[nextObjectID] = foundCentroids[obj]
                    nextObjectID+=1

            for row in usedRows:
                c = objects[row]
                cv2.putText(frame, 'ID '+str(row), (c[0]-20, c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.circle(frame, (c[0], c[1]), 4, (0,0,255), cv2.FILLED)

    cv2.imshow('frame',frame)

    if(cv2.waitKey(1)==ord("q")):
        break

cap.release()
cv2.destroyAllWindows()