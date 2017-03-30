import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    conv_stride = 1
    p = (filter_size-1)/2
    Hc = 1+(H-filter_size+2*p)/conv_stride
    Wc = 1+(W-filter_size+2*p)/conv_stride
    
    self.params['W1'] = np.random.normal(0,weight_scale,(num_filters,C,filter_size,filter_size))
    self.params['b1'] = np.zeros((num_filters,))
    print (num_filters,C,filter_size,filter_size)
    pool_width = 2
    pool_height = 2
    pool_stride = 2

    Hp = 1+(Hc-pool_height)/pool_stride
    Wp = 1+(Wc-pool_height)/pool_stride

    self.params['W2'] = np.random.normal(0,weight_scale,(num_filters*Hp*Wp,hidden_dim))
    self.params['b2'] = np.zeros((hidden_dim,))
    print (num_filters*Hp*Wp,hidden_dim)
    self.params['W3'] = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
    print (hidden_dim,num_classes)
    self.params['b3'] = np.zeros((num_classes,))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    #print np.sum(self.params['W1'])
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    X, cache_conv = conv_forward_im2col(X, W1, b1,conv_param)
    X, cache_relu1 = relu_forward(X)
    X,cache_pool = max_pool_forward_fast(X, pool_param)
    X_shape = X.shape
    X = X.reshape((X_shape[0],-1))
    X, cache_aff_hid = affine_forward(X,W2, b2)
    X, cache_relu2 = relu_forward(X)
    scores, cache_aff_out = affine_forward(X,W3, b3)
    #print np.sum(scores)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dx = softmax_loss(scores, y)
    #print np.sum(dx),np.sum(y)
    dx,grads['W3'],grads['b3'] = affine_backward(dx, cache_aff_out)
    dx = relu_backward(dx, cache_relu2)
    dx,grads['W2'],grads['b2'] = affine_backward(dx, cache_aff_hid)
    dx = dx.reshape((X_shape))
    dx = max_pool_backward_fast(dx, cache_pool)
    dx = relu_backward(dx, cache_relu1)
    dx, grads['W1'],grads['b1'] = conv_backward_im2col(dx,cache_conv)
    loss += 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))
    
    grads['W3'] += self.reg*W3
    grads['W2'] += self.reg*W2
    grads['W1'] += self.reg*W1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
class MyConv(object):
  ''' This is a custom network I've wriiten to classify images, gives the option of using batchnorm 
  and dropout, and also arbitrary number of conv,affine layers
  A network with arbitrary number of hidden units.
     [conv-relu-pool-{dropout}]XL - [affine-{batchnorm}-relu]XM -affine-softmax
     Input : numpy array of shape (N, C, H, W )
     Output: scores for each class'''

  def __init__(self,input_dim = (3, 32, 32), conv_filters=[10,20,30],filter_size=3,
               affine_dims=[200,200],num_classes=10, weight_scale=1e-3, reg=0.0,
                dtype=np.float64, dropout=0, use_batchnorm=True):
    
    
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.bn_params = {}
    self.dropout_params = {}
    self.use_dropout = False
    pad = (filter_size-1)/2
    pool_width = 2
    pool_height = 2
    pool_stride = 2

    C, H, W = input_dim
    conv_stride = 1
    filter_sizes = [C] + conv_filters

    self.filter_size = filter_size
    self.L = len(filter_sizes) 
    self.M = len(affine_dims)

    for i in range(self.L-1):
      weight = 'W'+str(i+1)
      bias = 'b'+str(i+1)

      self.params[weight] = np.random.normal(0,weight_scale,(filter_sizes[i+1],filter_sizes[i],filter_size,filter_size))
      self.params[bias] = np.zeros((filter_sizes[i+1],))

      if self.use_batchnorm:
        bn_param = 'bn_param'+str(i+1)
        self.bn_params[bn_param] = {'mode': 'train','running_mean': np.zeros(filter_sizes[i + 1]),'running_var': np.zeros(filter_sizes[i + 1])}
        
        gamma = 'gamma'+str(i+1)
        beta = 'beta' + str(i+1)
        
        self.params[gamma] = np.ones((filter_sizes[i + 1]))
        self.params[beta] = np.zeros((filter_sizes[i + 1]))

      Hc = 1+(H-filter_size+2*pad)/conv_stride
      Wc = 1+(W-filter_size+2*pad)/conv_stride

      H = 1+(Hc-pool_height)/pool_stride
      W = 1+(Wc-pool_height)/pool_stride

      if self.use_dropout:
        dropout_param = 'dropout_param'+str(i+1)
        self.dropout_params[dropout_param] = {'mode': 'train', 'p': dropout}
        

    affine_dims = [H*W*filter_sizes[-1]]+affine_dims
    
    for i in range(self.M):
      ind = self.L+i

      weight = 'W'+str(ind)
      bias = 'b'+str(ind)

      self.params[weight] = np.random.normal(0,weight_scale,(affine_dims[i],affine_dims[i+1]))
      self.params[bias] = np.zeros((affine_dims[i+1],))

      if self.use_batchnorm:
        bn_param = 'bn_param'+str(ind)
        self.bn_params[bn_param] = {'mode': 'train','running_mean': np.zeros(affine_dims[i + 1]),'running_var': np.zeros(affine_dims[i + 1])}
        
        gamma = 'gamma'+str(ind)
        beta = 'beta' + str(ind)

        self.params[gamma] = np.ones((affine_dims[i + 1]))
        self.params[beta] = np.zeros((affine_dims[i + 1]))
        
      if self.use_dropout:
        dropout_param = 'dropout_param'+str(ind)
        self.dropout_params[dropout_param] = {'mode': 'train', 'p': dropout}
      

    ind += 1
    weight = 'W'+str(ind)
    bias = 'b'+str(ind)

    self.params[weight] = np.random.normal(0,weight_scale,(affine_dims[-1],num_classes))
    self.params[bias] = np.zeros((num_classes,))

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    N = X.shape[0]

    filter_size = self.filter_size
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    if self.use_dropout:
      for param in self.dropout_params:
        self.dropout_params[param]['mode']=mode

    if self.use_batchnorm:
      for param in self.bn_params:
        self.bn_params[param]['mode']=mode

    scores = None
    loss=0
    grads={}
    cache_layer=[]
    for i in range(self.L-1):
      cache=[]
      weight = 'W'+str(i+1)
      bias = 'b'+str(i+1)
      weight = self.params[weight]
      #print np.sum(weight)
      bias = self.params[bias]
      X, c = conv_forward_im2col(X, weight, bias,conv_param)
      cache.append(c)
      
      if self.use_batchnorm:
        bn_param = 'bn_param'+str(i+1)
        bn_param= self.bn_params[bn_param]
        ngamma='gamma'+str(i+1)
        nbeta = 'beta'+str(i+1)
        gamma = self.params[ngamma]
        beta = self.params[nbeta]
        
        X, c = spatial_batchnorm_forward(X, gamma, beta, bn_param)
        cache.append(c)

      X, c = relu_forward(X)
      #print "second forward",np.sum(X)
      cache.append(c)

      X,c = max_pool_forward_fast(X, pool_param)
      #print "3rd forward",np.sum(X),X.shape
      cache.append(c)

      if self.use_dropout:
        dropout_param = 'dropout_param'+str(i+1)
        dropout_param = self.dropout_params[dropout_param]
        X,c = dropout_forward(X,dropout_param)
        cache.append(c)
      cache_layer.append(cache)

    X_shape = X.shape
    
    X = X.reshape((N,-1))

    for i in range(self.M):
      cache=[]
      ind = self.L+i

      weight = 'W'+str(ind)
      bias = 'b'+str(ind)

      weight = self.params[weight]
      bias = self.params[bias]
      X,c = affine_forward(X,weight,bias)
      #print "4th forward",np.sum(X)
      cache.append(c)
      

      if self.use_batchnorm:
        ngamma='gamma'+str(ind)
        nbeta = 'beta'+str(ind)
        bn_param = 'bn_param'+str(ind)
        bn_param= self.bn_params[bn_param]
        gamma = self.params[ngamma]
        beta = self.params[nbeta]
        X,c = batchnorm_forward(X, gamma, beta,bn_param)
        cache.append(c)
        
      X, c = relu_forward(X)
      #print "5th forward",np.sum(X),X.shape
      cache.append(c)
      cache_layer.append(cache)

    weight='W'+str(ind+1)
    bias='b'+str(ind+1)
    weight = self.params[weight]
    bias = self.params[bias]
    scores,cache = affine_forward(X,weight,bias)
    #print "result",scores
    #print np.sum(scores)

    #print "==========================================================="
    if y is None:
      return scores

    reg_loss =0
    loss, dx = softmax_loss(scores, y)
    #print loss
    #print np.sum(dx),np.sum(y)
    dx, dw, db = affine_backward(dx, cache)
    weight = 'W' + str(self.M+self.L)
    bias = 'b'+ str(self.M+self.L)
    grads[weight] = (dw+(self.reg*self.params[weight]))
    reg_loss += 0.5*self.reg*np.sum(self.params[weight]*self.params[weight])
    grads[bias] = db

    
    for i in range(1,self.M+1):
      j = -1
      cache = cache_layer[-i]
      dx = relu_backward(dx, cache[j])

      j-=1

      if self.use_batchnorm:
        
        gamma='gamma'+str(self.L+self.M-i)
        beta='beta'+str(self.L+self.M-i)
        dx, dgamma, dbeta = batchnorm_backward_alt(dx, cache[j])
        grads[gamma] = dgamma
        grads[beta] = dbeta
        j -=1
      
      weight = 'W'+str(self.L+self.M-i)
      bias = 'b'+str(self.L+self.M-i)
      dx,dw,db = affine_backward(dx, cache[j])
      reg_loss += 0.5*self.reg*np.sum(self.params[weight]*self.params[weight])
      grads[weight] = (dw+(self.reg*self.params[weight]))
      grads[bias] = db


    dx = dx.reshape((X_shape))
    
    for i in range(self.L-1):
      j =-1
      cache = cache_layer[-(self.M+i+1)]

      if self.use_dropout:
        dx = dropout_backward(dx, cache[j]) 
        j -= 1
        
      dx = max_pool_backward_fast(dx, cache[j])
      j -=1
      dx = relu_backward(dx, cache[j])
      j -=1
      if self.use_batchnorm:
        ngamma='gamma'+str(self.L-1-i)
        nbeta = 'beta'+str(self.L-1-i)
        dx, dgamma, dbeta = spatial_batchnorm_backward(dx, cache[j])
        grads[ngamma] = dgamma
        grads[nbeta] = dbeta
        j -=1
        
      weight = 'W' + str(self.L-i-1)
      bias = 'b'+ str(self.L-i-1)
      dx, dw,grads[bias] = conv_backward_im2col(dx,cache[j])
      grads[weight] = (dw+(self.reg*self.params[weight]))
      reg_loss += 0.5*self.reg*np.sum(self.params[weight]*self.params[weight])
      
    loss += reg_loss
    return loss,grads
