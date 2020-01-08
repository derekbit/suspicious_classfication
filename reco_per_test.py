#import theano
import numpy as np
import pandas as pd
import cv2
import os
import h5py
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model, Sequential
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.io_utils import HDF5Matrix

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

DIR='/home/dereksu/keras/dataset'
MIN_SEQ_LEN=600
MAX_SEQ_LEN=1000000
SUBSAMPLE=6

REQUIRED_LABELS = ['normal', 'abnormal']

def preprocess_frame(frame):
  frame = cv2.resize(frame, (299, 299))
  return preprocess_input(frame)


def encode_video(path):
  # Create base model
  base_model = InceptionV3(weights='imagenet', include_top=True)

  # Will extrace features at the final pooling layers
  model = Model(
    inputs = base_model.input,
    outputs = base_model.get_layer('avg_pool').output
  )

  cap = cv2.VideoCapture(path)

  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  num_groups = int(frame_count / MIN_SEQ_LEN)

  features = []

  for i in range(num_groups):
    frames = []

    for j in (range(MIN_SEQ_LEN * i, MIN_SEQ_LEN * (i + 1))):
      if j % SUBSAMPLE == 0:
        ret, frame = cap.read()
        frame = preprocess_frame(frame)
        frames.append(frame)

    feature = model.predict(np.array(frames))
    features.append(feature)

    del frames[:]

  return np.array(features)


video_path = '/home/dereksu/keras/dataset/normal/Normal_Videos_196_x264.mp4'
model_path = '/home/dereksu/keras/models/3/Activity_Recognition.h5'
pretrain_model = load_model(model_path)
features = encode_video(video_path)

print('features shape: ', features.shape)

predictions = pretrain_model.predict_classes(features)
print(predictions)

if sum(predictions) != 0:
  print('suspicious event')
