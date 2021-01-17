import tensorflow as tf
import cv2
import numpy as np

labls = np.load('data/label.npy', allow_pickle=True) 
imgs = np.load('data/data.npy',allow_pickle=True)