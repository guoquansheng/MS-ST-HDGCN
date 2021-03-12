import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from os.path import join as pjoin
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras
import numpy as np
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt




config = tf.ConfigProto()  
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--n_route', type=int, default=228)
parser.add_argument('--n_his', type=int, default=16) 
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=50)  
parser.add_argument('--epoch', type=int, default=50)  
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)  
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')


args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt

blocks = [[1, 32, 64], [64, 32, 128], [128, 32, 128]]    
if args.graph == 'default':     
    W = weight_matrix(pjoin('./dataset', 'W_228.csv'))     
else:
    W = weight_matrix(pjoin('./dataset', args.graph))


L = scaled_laplacian(W)
Lk = cheb_poly_approx(L, Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))    # compat.v1.
data_file = f'V_228.csv'   
n_train, n_val, n_test = 34, 5, 5
PeMS = data_gen(pjoin('./dataset', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')



if __name__ == '__main__':
    model_train(PeMS, blocks, args)
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)