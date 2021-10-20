import os
import tensorflow as tf
from problem import problem
from pdb import set_trace
import numpy as np
from ampistanet_reduced import AMPNet
# from netcomplex import AMPNet
import admmnet3

def parse(x):
  words = []
  words.append('')
  for i in x:
      if i != '_':
        words[-1] += i
      elif i == '_':
        words.append('')
  return words

def parse_dims(x):
  words = []
  words.append('')
  for i in x:
      if i != 'x':
        words[-1] += i
      elif i == 'x':
        words.append('')
  return words

def parse_SNR(x):
  word = ''
  for i in x:
      if i in ['S','N','R','d','B','=']:
        pass
      else:
        word += i
  return word

def get_path(params):
  N,L,M,Ng,K,Ntxrx,J,SNR = params
  X = os.listdir('./nets/results')
  # X = os.listdir('/Users/Jeremy/Documents/GitHub/mmv_cvx/nets/results')
  # X = os.listdir('/Users/Jeremy/Documents/mmv-old/dunn/nets/results')
  # X = os.listdir('C:/Users/Jeremy/Documents/results_admm_ampista_3_25_21')
  # X = os.listdir('/Users/Jeremy/Documents/mmv/isit_results/')
  res = []
  for x in X:
    p = parse(x)
    # print(p)
    if len(p) == 12:
    
      d = parse_dims(p[1])
      if d[0] == str(N) and \
         d[1] == str(M) and \
         d[2] == str(L) and \
         p[2][-1] == str(J) and \
         p[3][-1] == str(K) and \
         parse_SNR(p[4]) == str(SNR):
         res.append(x)

  return res
    # return

def get_numlayers(x):
  res = ''
  for i in x:
    if i == 'l':
      return res
    res += i

def get_final_epoch(path):
  contents = os.listdir(path)
  for c in contents:
    p = parse(c)
    if p[0] == 'Epoch=30':
      return c

def loadnet(net, weights_path):
  obj = net.load_weights(weights_path)
  obj.expect_partial()
  return net

def gen_net(method, p, n):
  # tf.keras.backend.set_floatx('float64')
  if method == 'admm3':
    print('ADMM3 Net')
    p.sigma, p.mu, p.rho, p.taux, p.tauz = 6.9e-01, 1.0e-02, 2.0e+00, 1.4e-01, 7.2e-01
    tf.keras.backend.set_floatx('float32')
    return admmnet3.ADMMNet(p, n)
  elif method == 'ampista':
    print('AMP-ISTA Net')
    p.lam = .1
    p.damp_init = 0.6
    tf.keras.backend.set_floatx('float32')
    num_stages = {'amp':n, 'ista':5}
    return AMPNet(p, num_stages)
