import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("dataset.csv")

patterns = df["pattern"].values
tags = df["tag"].values

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns).toarray()

encoder = LabelEncoder()
y = encoder.fit_transform(tags)
y = np.array(y)

model = Sequential([
    Dense(64, activation="relu", input_shape=(X.shape[1],)),
    Dense(32, activation="relu"),
    Dense(len(np.unique(y)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X, y, epochs=50, batch_size=4)

model.save("dl_model.h5")
pickle.dump(encoder, open("label_encoder.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Deep Learning model trained successfully!")
