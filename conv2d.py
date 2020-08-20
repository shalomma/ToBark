import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical

import librosa
import librosa.display

df = pd.read_csv("../UrbanSound8K/UrbanSound8K.csv")
print(df.head())

feature = []
label = []


def parser():
    # Function to load files and extract features
    for i in range(8732):
        file_name = 'UrbanSound8K/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # We extract mfcc feature from data
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        feature.append(mels)
        label.append(df["classID"][i])
    return [feature, label]


temp = parser()
temp = np.array(temp)
data = temp.transpose()

X_ = data[:, 0]
Y = data[:, 1]
print(X_.shape, Y.shape)
X = np.empty([8732, 128])
for i in range(8732):
    X[i] = (X_[i])
Y = to_categorical(Y)
print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
X_train = X_train.reshape(6549, 16, 8, 1)
X_test = X_test.reshape(2183, 16, 8, 1)

input_dim = (16, 8, 1)

model = Sequential()
model.add(Conv2D(64, (3, 3), padding="same", activation="tanh", input_shape=input_dim))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding="same", activation="tanh"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation="tanh"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=90, batch_size=50, validation_data=(X_test, Y_test))
print(model.summary())

predictions = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print(score)

preds = np.argmax(predictions, axis=1)
result = pd.DataFrame(preds)
result.to_csv("UrbanSound8kResults.csv")
