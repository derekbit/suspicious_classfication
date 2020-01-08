#!/bin/bash

export THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32; python3 reco_train.py
