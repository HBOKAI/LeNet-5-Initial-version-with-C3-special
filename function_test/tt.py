import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
import numpy as np
import matplotlib.pyplot as plt
import os


outest = np.arange(36)
outest = outest.reshape(1,3,3,4)
weight = np.arange(144).reshape(3,3,4,4)
outest = tf.convert_to_tensor(outest)
weight = tf.convert_to_tensor(weight)
# print(tf.nn.conv2d(outest,weight,[1,1,1,1],'VALID'))
# print("Test start: \n")
# print(tf.nn.conv2d(outest[...,0:1],weight[...,0:1,0:1],[1,1,1,1],'VALID'))
# print(tf.nn.conv2d(outest[...,1:2],weight[...,1:2,0:1],[1,1,1,1],'VALID'))
# print(tf.nn.conv2d(outest[...,2:3],weight[...,2:3,0:1],[1,1,1,1],'VALID'))
# print(tf.nn.conv2d(outest[...,3:4],weight[...,3:4,0:1],[1,1,1,1],'VALID'))

# print("All",weight)
# print("o:1",weight[...,0:1,0:1])
print(outest)
print(outest[...,0:1]+outest[...,1:2])
a = outest[...,0:1]+outest[...,1:2]
# a = tf.add(outest[...,0:1],outest[...,1:2])
outest = tf.concat(axis=3,values=[outest,a])
print(outest[...,4:5])
print(outest.shape)
# print(outest[...,0]+outest[...,1])

# print(tf.add(outest[...,0:1],outest[...,1:2]))