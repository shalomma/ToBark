import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical

import librosa
import librosa.display

df = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
print(df.head())

N = 8732

feature = []
label = []


def parser():
    # Function to load files and extract features
    for i in range(N):
        file_name = 'UrbanSound8K/audio/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # We extract mfcc feature from data
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        feature.append(mels)
        label.append(df["classID"][i])
    return np.array(feature), np.array(label)


data, labels = parser()

torch.save(torch.tensor(data), 'mel_data.pt')
torch.save(torch.tensor(data), 'mel_labels.pt')

X = data.reshape((-1, 16, 8, 1))
Y = to_categorical(labels)
print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)


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
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=90, batch_size=50, validation_data=(X_test, Y_test))
