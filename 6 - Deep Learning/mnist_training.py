import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.utils import to_categorical


# Carichiamo il Dataset
DATASET_PATH = "../datasets/mnist/mnist.csv"
dataset = np.loadtxt(open(DATASET_PATH, "rb"), delimiter=",")

# Prepariamo i Dati
X = dataset[:,:-1]
y = dataset[:,-1:]

X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

X_train=X_train/255
X_test=X_test/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Creiamo la Rete Neurale
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(256, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=30, batch_size=512)

# Valutiamo il Modello
metrics_train = model.evaluate(X_train, y_train, verbose=0)
metrics_test = model.evaluate(X_test, y_test, verbose=0)

print("Train Accuracy = %.4f - Train Loss = %.4f" % (metrics_train[1], metrics_train[0]))
print("Test Accuracy = %.4f - Test Loss = %.4f" % (metrics_test[1], metrics_test[0]))

# Esportiamo il Modello
model.save("model_mnist.h5")