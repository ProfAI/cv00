import cv2
from tensorflow.keras.models import load_model

SCALE = (28, 28)

# Importiamo il Modello
model = load_model('model_mnist.h5')

# Carichiamo l'immagine
img_path = input("Inserire il percorso all'immagine: ")
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Processiamo l'immagine
small_img = cv2.resize(img, SCALE)
x = small_img.flatten().astype(float)
x/=255.

# Riconoscimento dell'immagine
x = x.reshape(1,x.shape[0])
proba = model.predict(x)
y = model.predict_classes(x)
y = y[0]
proba = proba[0]

print(proba)
print(y)

print("Network prediction: %d (%s)" % (y, proba))