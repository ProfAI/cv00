import cv2
import joblib

SCALE = (28, 28)

# Importiamo il Modello
lr = joblib.load("binary_mnist_model.pkl")

# Carichiamo l'immagine
img_path = input("Inserire il percorso all'immagine: ")
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Processiamo l'immagine
small_img = cv2.resize(img, SCALE)
x = small_img.flatten().astype(float)
x/=255.

# Riconoscimento dell'immagine
x = x.reshape(1,x.shape[0])
y = lr.predict(x)
proba = lr.predict_proba(x)

print("Model prediction: %d (%s)" % (y[0],proba[0]))