import numpy as np
from PIL import Image

arr = np.random.randint(0, high=255, size=(100,100)) # creiamo un'immagine casuale in B/W

# Mostriamo l'immagine
img = Image.fromarray(arr)
img.show()

print("Prima riga")
print(arr[0])

print("Dalla seconda alla quinta riga")
print(arr[1:5])

print("Ultima riga")
print(arr[-1])

print("Prima colonna")
print(arr[:,0])

print("Dalla seconda alla quinta colonna")
print(arr[:,1:5])

print("Ultima colonna")
print(arr[:,-1])

print("Seconda colonna dell'ultima riga")
print(arr[-1,1])

# Creiamo un contorno nero
arr[0] = np.zeros((100))
arr[-1] = np.zeros((100))
arr[:,0] = np.zeros((100)) 
arr[:,-1] = np.zeros((100))

img = Image.fromarray(arr)
img.show()

#creiamo un quadrato bianco
np_white = np.ones((20,20))*255
x_offset = int(np_white.shape[0]/2)
y_offset = int(np_white.shape[1]/2)
x_start = int(arr.shape[0]/2-x_offset)
x_end = int(arr.shape[0]/2+x_offset)
y_start = int(arr.shape[1]/2-y_offset)
y_end = int(arr.shape[1]/2+y_offset)
arr[x_start:x_end,y_start:y_end] = np_white
img = Image.fromarray(arr)
img.show()