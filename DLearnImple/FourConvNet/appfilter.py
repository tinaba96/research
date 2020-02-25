import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from FourConvNet import FourConvNet
from matplotlib.image import imread
from common.layers import Convolution

def filter_show(filter, nx=4, show_num=16):
  FN, C, FH, FW = filter.shape
  ny = int(np.ceil(show_num / nx))

  fig = plt.figure()
  fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

  for i in range(show_num):
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filter[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')

network = FourConvNet(input_dim=(1, 28,28), conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, hidden_size=100, output_size=10, weight_init_std=0.01)

#学習後の重み
network.load_params("params.pkl")

filter_show(network.params['W1'], 16)
#filter_show(network.params['W2'], 16)

img = imread('../dataset/lena_gray.png')
img = img.reshape(1, 1, *img.shape)

fig = plt.figure()

w_idx = 1

for i in range(16):
  w = network.params['W1'][i]
  b = 0

  w = w.reshape(1, *w.shape)
  #b = b.reshape(1, *b.shape)
  conv_layer = Convolution(w,b)
  out = conv_layer.forward(img)
  out = out.reshape(out.shape[2], out.shape[3])

  ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
  ax.imshow(out, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()





