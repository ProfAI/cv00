import numpy as np
from PIL import Image

img = Image.open("res/cat.jpg")
img.show()

arr = np.array(img)
print(arr.shape)

print(arr[:,:,2])

