import sys, os



class FourConvNet:

  '''
  conv1-relu1-pool1-conv2-relu2-pool2-affine1-relu3-affine2-softmax
  '''
  def__init__(self, input_dim(1, 28, 28), conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},hidden_size=100, output_size= 10, weight_init_std=0.01):
    filter_num = conv_param['filter_num']
    filiter_size = conv_param['filter_size']
    filter_pad = conv_param['pad']
    filter_size = input_dim[1]
    conv_output_sixe = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
    pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    self.params['b1'] = np.zeros(filter_num)
    self.params['W2'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    self.params['b2'] = np.zeros(filter_num)
    self.params['W3'] = weight_init_std * np.random.randn(pool_output_size, hidden_size))
    self.params['b3'] = np.zeros(hidden_size)
    self.params['W4'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b4'] = np.zeros(output_size)

    #レイヤの生成
    self.layers = OrderedDict()
    self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
    self.layers['Relu1'] = Relu()
    self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
    self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param['stride'], conv_param['pad'])
    self.layers['Relu2'] = Relu()
    self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
    self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
    self.layers['Relu3'] = Relu()
    self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

    self.last_layer = SoftmaxWithLoss()

def predict(self, x):
  for layer in self.layers.values():
    x = layer.forward(x)
  return x

def loss(self, x, t):





