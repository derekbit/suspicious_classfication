import theano
import numpy as np
import pandas as pd
import cv2
import os
import h5py
import time
from tqdm import tqdm                                                                                                                                                       
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.utils.io_utils import HDF5Matrix
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DIR='/home/dereksu/keras/dataset'
MIN_SEQ_LEN=600
MAX_SEQ_LEN=1000000
SUBSAMPLE=6

BATCH_SIZE=32
EPOCHS=1500

REQUIRED_LABELS = ['normal', 'abnormal']

def get_dataset(path):
  names = ['label', 'video_name', 'frames', 'fps']
  return pd.read_csv(path, names = names)

def clean_dataset(dataset):
  mask = np.logical_and(dataset['frames'] >= MIN_SEQ_LEN, dataset['frames'] <= MAX_SEQ_LEN)
  return dataset[mask]

def get_label_dict(dataset):
  labels = list(dataset['label'].unique())
  indices = np.arange(0, len(labels))

  return dict(zip(labels, indices))

def split_dataset(dataset, train_size=0.75):
  label = (dataset.groupby(['label']))
  un = dataset['label'].unique()

  normal = label.get_group(un[0])
  abnormal = label.get_group(un[1])

  normal = shuffle(normal)
  abnormal = shuffle(abnormal)

  train = normal.iloc[:int(normal.shape[0] * train_size)]
  df = abnormal.iloc[:int(abnormal.shape[0] * train_size)]
  train = train.append(df)

  test = normal.iloc[int(normal.shape[0] * train_size):]
  df = abnormal.iloc[int(abnormal.shape[0] * train_size):]
  test = test.append(df)

  return (train, test)

def preprocess_frame(frame):
  frame = cv2.resize(frame, (299, 299))
  return preprocess_input(frame)


def encode_video(row, model, input_f, output_y, label_index):
  path = os.path.join(DIR, str(row["label"].iloc[0]) ,str(row["video_name"].iloc[0]))

  cap = cv2.VideoCapture(path)

  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  num_groups = int(frame_count / MIN_SEQ_LEN)

  if num_groups < 10:
    return

  for i in range(num_groups):
    if i != 10:
      continue

    print("label: %s" % str(row["label"].iloc[0]))
    frames = []

    for j in (range(MIN_SEQ_LEN * i, MIN_SEQ_LEN * (i + 1))):
      if j % SUBSAMPLE == 0:
        ret, frame = cap.read()
        frame = preprocess_frame(frame)
        frames.append(frame)

    features = model.predict(np.array(frames))
    index = label_index[row['label'].iloc[0]]

    input_f.append(features)
    output_y.append(index)

    del frames

def encode_dataset(dataset, model, label_index, phase):
  input_f = []
  output_y = []

  for i in tqdm(range(dataset.shape[0])):
    label = str(dataset.iloc[[i]]['label'].iloc[0])
    if label in REQUIRED_LABELS:
      encode_video(dataset.iloc[[i]], model, input_f, output_y, label_index)

  output_labels = np_utils.to_categorical(output_y)
  return (np.array(input_f), np.array(output_labels))
  
def extract_features():
  # Create base model
  base_model = InceptionV3(weights='imagenet', include_top=True)

  # Will extrace features at the final pooling layers
  model = Model(
    inputs = base_model.input,
    outputs = base_model.get_layer('avg_pool').output
  )

  dataset = get_dataset(DIR + '/' + 'data_file.csv')
  dataset = clean_dataset(dataset)
  label_index = get_label_dict(dataset)

  train, test = split_dataset(dataset, train_size = 0.75)

  #print("train size: ", train.shape)
  #print("test size: ", test.shape)

  (x_test, y_test) = encode_dataset(test, model, label_index, 'test')
  return (x_test, y_test)

(x_test, y_test) = extract_features()

print(x_test.shape)
print(y_test.shape)

pretrain_model = load_model('/home/dereksu/keras/models/3/Activity_Recognition.h5')
predictions = pretrain_model.predict_classes(x_test)
print(predictions)
scores = pretrain_model.evaluate(x_test, y_test)
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))
