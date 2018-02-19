import numpy as np
import scipy.io as spio
d =spio.loadmat('/home/saurabh/scratch/saved_data/loaddata_rest.mat')
data = d['a']
d =spio.loadmat('/home/saurabh/scratch/saved_data/slabels_rest.mat')
slabels = d['s']
slabels= slabels.T
d =spio.loadmat('/home/saurabh/scratch/saved_data/tlabels_rest.mat')
tlabels = d['t']
tlabels= tlabels.T
d =spio.loadmat('/home/saurabh/scratch/saved_data/mlabels_rest.mat')
mlabels = d['m']
mlabels= mlabels.T

d =spio.loadmat('/home/saurabh/scratch/saved_data/sid.mat')
sid2 = d['s']

data2 = data
idx = tlabels * np.asarray(mlabels)

exec(open('/home/saurabh/scratch/code/lstm_tf.py').read())
