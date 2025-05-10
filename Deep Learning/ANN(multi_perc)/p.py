import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf


# Load dataset
dataset = pd.read_csv(r"C:\Users\vishw\Downloads\credit_score_data.csv")

# Split features and target
x = dataset.iloc[:, :-1]  # Features
y = dataset.iloc[:, -1]   # Target

# Normalize features
sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(x), columns=x.columns)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)

# Print training shape
print("Training shape:", x_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import BatchNormalization

# Build ANN model
# ann = Sequential()
# ann.add(Dense(5, input_dim=x_train.shape[1], activation="relu"))  # ✅ Fixed input_dim
# # ann.add(BatchNormalization())
# ann.add(Dropout(0.2))
# ann.add(Dense(3, activation="relu"))
# # ann.add(BatchNormalization())
# ann.add(Dropout(0.2))
# ann.add(Dense(2, activation="relu"))
# ann.add(Dropout(0.2))
# ann.add(Dense(1, activation="sigmoid"))

ann = Sequential()
ann.add(Dense(64, input_dim=x_train.shape[1], activation="relu"))  
ann.add(Dropout(0.3))
ann.add(Dense(32, activation="relu"))
ann.add(Dropout(0.3))
ann.add(Dense(16, activation="relu"))
ann.add(Dropout(0.3))
ann.add(Dense(1, activation="sigmoid"))


# Compile model
ann.compile(optimizer="nadam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
history = ann.fit(x_train, y_train, batch_size=16, epochs=100, verbose=1)
print(history)

# Predict
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5).ravel()  # ✅ Fixed shape issue

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(" Testing Accuracy is:", accuracy)

train_accuracy = history.history['accuracy'][-1]  # Last recorded accuracy
print("Final Training Accuracy:", train_accuracy * 100, "%")