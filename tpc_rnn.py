import numpy as np
import random
import os
import csv
import json
import time
import shutil
import json
from random import shuffle

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.utils import to_categorical

import tensorflowjs as tfjs

HOME_DIR = os.path.expanduser("~/")
INPUT_FOLDER = HOME_DIR + "Desktop/stupid/collector/data"
MOVES = ['tob', 'pae', 'charge']
EPOCHES = 15
feature_length = 6

print("Loading inputs")
all_data = []
for move in MOVES:
  categorical_move = to_categorical(MOVES.index(move), len(MOVES))
  json_files = [ INPUT_FOLDER + "/" + move + "/" + filename for filename in os.listdir(INPUT_FOLDER + "/" + move) if filename.endswith(".json") ]
  for path in json_files:
    with open(path, 'rb') as f:
      # Assume that gyro always follow accel. If gyro is missing, then the event is missing, so use previous event
      prev_accel = None
      prev_gyro = None
      prev_type = None
      data = []
      for line in f:
        current = json.loads(line)
        # We always want current gyro, but sometimes want previous accel
        if current['type'] == 'gyro':
          prev_gyro = current
        if prev_type == 'accel' and prev_gyro:
          data.append([prev_accel['x'], prev_accel['y'], prev_accel['z'],
                       prev_gyro['x'], prev_gyro['y'], prev_gyro['z']])
        prev_type = current['type']
        if current['type'] == 'accel':
          prev_accel = current

      # Only use every other element to improve speed
      all_data.append((data[1::2], categorical_move))

# Superstition!
random.shuffle(all_data)
random.shuffle(all_data)

print("Training")
train_data = all_data[0:len(all_data)-3]
test_data = all_data[len(all_data)-3:]
model = Sequential()
model.add(Dropout(0.2, input_shape=(None, feature_length)))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(len(MOVES), activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

for epoch in range(EPOCHES):
  for i, (x, y) in enumerate(train_data):
    print("Epoch %d Item %d" % (epoch, i))
    model.fit(np.array([x]), np.array([y]), batch_size=1)
    
# Evaluate
for i, (x, y) in enumerate(test_data):
  prediction = model.predict(np.array([x]))
  print(prediction)
  print("Predict: " + MOVES[np.argmax(prediction)] + " Actual: " + MOVES[np.argmax(y)])

model.save('model.h5')
tfjs.converters.save_keras_model(model, 'model.tfjs')