# Models/1
  input_shape = (int(MIN_SEQ_LEN / SUBSAMPLE), 2048)

  model.add(LSTM(1024, input_shape = input_shape, dropout=0.5))
  model.add(Dropout(0.5))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(16, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(2, activation='softmax'))


984/984 [==============================] - 4s 4ms/step - loss: 0.3077 - accuracy: 0.8709 - val_loss: 0.5208 - val_accuracy: 0.8524
984/984 [==============================] - 2s 2ms/step
        [Info] Accuracy of training data = 90.9%
210/210 [==============================] - 0s 2ms/step
        [Info] Accuracy of testing data = 85.2%

# Models/2

  input_shape = (int(MIN_SEQ_LEN / SUBSAMPLE), 2048)

  model.add(LSTM(2048, input_shape = input_shape, dropout=0.5))
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='softmax'))


984/984 [==============================] - 10s 10ms/step - loss: 0.2292 - accuracy: 0.9096 - val_loss: 0.5117 - val_accuracy: 0.8571
984/984 [==============================] - 3s 3ms/step
        [Info] Accuracy of training data = 94.3%
210/210 [==============================] - 1s 3ms/step
        [Info] Accuracy of testing data = 85.7%

# Models/3

  input_shape = (int(MIN_SEQ_LEN / SUBSAMPLE), 2048)

  model = Sequential()
  model.add(LSTM(2048, input_shape = input_shape, dropout=0.5))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='softmax'))


984/984 [==============================] - 10s 10ms/step - loss: 0.1859 - accuracy: 0.9278 - val_loss: 0.5382 - val_accuracy: 0.8667
984/984 [==============================] - 3s 3ms/step
	[Info] Accuracy of training data = 96.2%
210/210 [==============================] - 1s 3ms/step
	[Info] Accuracy of testing data = 86.7%
