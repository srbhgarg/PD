import numpy as np
import tensorflow as tf  #TF 1.1.0rc1
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.contrib.rnn import LSTMCell
import logging
import sys
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
        E.g. for use with categorical_crossentropy.
          Arguments:
      y: class vector to be converted into a matrix
           (integers from 0 to num_classes).
         num_classes: total number of classes.
     Returns:
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


import scipy.io as spio
d =spio.loadmat('/home/saurabh/scratch/saved_data/loaddata_rest90d_wo_minmax.mat')
data = d['a'] # X 
d =spio.loadmat('/home/saurabh/scratch/saved_data/slabels_rest90d_wo_minmax.mat')
slabels = d['s'] # subject.. PD vs. HC
slabels= slabels.T
d =spio.loadmat('/home/saurabh/scratch/saved_data/tlabels_rest90d_wo_minmax.mat')
tlabels = d['t'] # task
tlabels= tlabels.T
d =spio.loadmat('/home/saurabh/scratch/saved_data/mlabels_rest90d_wo_minmax.mat')
mlabels = d['m'] # medication
mlabels= mlabels.T

d =spio.loadmat('/home/saurabh/scratch/saved_data/sid90d_wo_minmax.mat')
sid2 = d['m']


#slabels = Resting or PD
#mlabels = onmed vs offmed
#tlabels = resing vs task
#organize data
# pick resting state and onmed
idx = tlabels * np.asarray(mlabels)
data = data[idx[:,0]==1] #data2[idx ...
sid = sid2[:,idx[:,0]==1]

slabels = slabels[idx[:,0]==1,:]
#reshape data to batch x time x features
#inputs argument, where the dimensions are interpreted by default as batch_size x num_timesteps x num_features.
#(If you pass time_major=True, they are interpreted as num_timesteps x batch_size x num_features.)
data = np.transpose(data, axes=(0,2,1)) #default
#data = np.transpose(data, axes=(2,0,1)) #for time_major= True

uSIDS = np.unique(sid)
nsubjs=len(uSIDS)
n=data.shape[0]
srate = 15 

all_sinds = np.arange( 0, nsubjs, 1)           # indices to whole dataset
all_sinds = uSIDS[all_sinds]
test_sinds  = np.arange( 1, nsubjs, 15)   # take out every k-th subject as test 
test_sinds = uSIDS[test_sinds]
train_sinds = np.setdiff1d( all_sinds, test_sinds) 
valid_sinds = train_sinds[::srate] #sample the training subset to get validation set
train_sinds = np.setdiff1d( train_sinds, valid_sinds)

L=[];
#SUBJECT_IDS start from 1 to N where as indices start from 0 to N-1
for p in test_sinds:
    pp = ( p == sid ).nonzero( )
    #L.append(pp[1])
    L= np.concatenate( (L,pp[1]) , axis=0 )  
L = L.astype(int)
test_inds = L;
tr=[];
for p in train_sinds:
    pp = ( p == sid ).nonzero( ) 
    tr= np.concatenate( (tr,pp[1]) , axis=0 )  
tr = tr.astype( int )
train_inds = tr;
v=[];
for p in valid_sinds:
    pp = ( p == sid ).nonzero( ) 
    v= np.concatenate( (v,pp[1]) , axis=0 )  
v = v.astype( int )
valid_inds = v;

train_X = data[train_inds]
train_y =np.asarray(slabels)[train_inds]
test_X = data[test_inds]
test_y =to_categorical(np.asarray(slabels)[test_inds],2)
valid_X = data[valid_inds]
valid_y = to_categorical(np.asarray(slabels)[valid_inds],2)

num_classes = len(np.unique(train_y))
train_y = to_categorical(np.asarray(slabels)[train_inds],2)

left_rois = np.concatenate((np.arange(0,18),np.arange(39,115),np.array([189,191,193,195,197,199]) ))
right_rois = np.concatenate((np.arange(18,39),np.arange(115,188),np.array([188,190,192,194,196,198]) ))
del data

"""Hyperparamaters"""
#left
seed_l = int( sys.argv[1] )# 8  # or any number
batch_size_l = int( sys.argv[2] ) #256
dout_l = float( sys.argv[3] ) #0.7
num_layers_l = int( sys.argv[4] )#3
hidden_size_l = int( sys.argv[5] )#16 
#learning_rate=0.00001
epoch_count_l =int( sys.argv[6] ) #3700
#beta value = 0.8 to 0.999
beta_l = float( sys.argv[7] )#0.9
optim_l =1  # 1 for Adam otherwise RMS
steps_l = int( sys.argv[8] )#100
np.random.seed(seed_l)
tf.set_random_seed(seed_l)
starter_learning_rate_l = float(sys.argv[9]) #0.00001
decay_rate_l = float(sys.argv[10]) #0.96

#right
seed_r = int( sys.argv[11] )# 8  # or any number
batch_size_r = int( sys.argv[12] ) #256
dout_r = float( sys.argv[13] ) #0.7
num_layers_r = int( sys.argv[14] )#3
hidden_size_r = int( sys.argv[15] )#16 
#learning_rate=0.00001
epoch_count_r =int( sys.argv[16] ) #3700
#beta value = 0.8 to 0.999
beta_r = float( sys.argv[17] )#0.9
optim_r =1  # 1 for Adam otherwise RMS
steps_r = int( sys.argv[18] )#100
np.random.seed(seed_r)
tf.set_random_seed(seed_r)
starter_learning_rate_r = float(sys.argv[19]) #0.00001
decay_rate_r = float(sys.argv[20]) #0.96

# Define Model
dropout_l = tf.placeholder(tf.float32)
dropout_r = tf.placeholder(tf.float32)


global_step = tf.Variable(0, trainable=False)
learning_rate_l = tf.train.exponential_decay(starter_learning_rate_l, global_step,
                                                   steps_l, decay_rate_l, staircase=True)
learning_rate_r = tf.train.exponential_decay(starter_learning_rate_r, global_step,
                                                   steps_r, decay_rate_r, staircase=True)
logfile = "/home/saurabh/logs/log_Adam_%d_%d_%.3f_%d_%d_%d_%.3f_%d_%.8f_%.3f.log" %(seed_l, batch_size_l,dout_l, num_layers_l,hidden_size_l, epoch_count_l, beta_l, steps_l, starter_learning_rate_l, decay_rate_l  )
misfile = "/home/saurabh/scratch/saved_data/Adam_%d_%d_%.3f_%d_%d_%d_%.3f_%d_%.8f_%.3f.mat" %(seed_l, batch_size_l,dout_l, num_layers_l,hidden_size_l, epoch_count_l, beta_l, steps_l, starter_learning_rate_l, decay_rate_l  )
logging.basicConfig(filename=logfile,level=logging.DEBUG)

sessfile_l = "/home/saurabh/models/left_Adam_%d_%d_%.3f_%d_%d_%d_%.3f_%d_%.8f_%.3f.ckpt" %(seed_l, batch_size_l,dout_l, num_layers_l,hidden_size_l, epoch_count_l, beta_l, steps_l, starter_learning_rate_l, decay_rate_l  )
sessfile_r = "/home/saurabh/models/right_Adam_%d_%d_%.3f_%d_%d_%d_%.3f_%d_%.8f_%.3f.ckpt" %(seed_r, batch_size_r,dout_r, num_layers_r,hidden_size_r, epoch_count_r, beta_r, steps_r, starter_learning_rate_r, decay_rate_r  )
# We can put initializer= in the lstmcell 
#initializer = tf.random_uniform_initializer(-1, 1)#
def lstm_cell_l():
   return tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_size_l),output_keep_prob=dropout_l)
   #return tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_size, initializer = tf.contrib.layers.xavier_initializer(seed=0)),output_keep_prob=dropout)

def lstm_cell_r():
   return tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_size_r),output_keep_prob=dropout_r)

lstm_l = tf.contrib.rnn.MultiRNNCell([lstm_cell_l() for _ in range(num_layers_l)])
lstm_r = tf.contrib.rnn.MultiRNNCell([lstm_cell_r() for _ in range(num_layers_r)])

# data = Batch size x time steps x left features
target_l= tf.placeholder(tf.float32, [None, num_classes])
inputs_l = tf.placeholder(tf.float32, [None, 90, 100])
with tf.variable_scope('left'):
    outputs_l, _ = tf.nn.dynamic_rnn(lstm_l, inputs_l, dtype=tf.float32)


#take the last output
#output = outputs[-1]
output_l = tf.transpose(outputs_l, [1, 0, 2])
last_l = tf.gather(output_l, int(output_l.get_shape()[0]) - 1)
#softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
#softmax_b = tf.get_variable("softmax_b", [num_classes])
softmax_w_l = tf.Variable(tf.truncated_normal([hidden_size_l, num_classes], stddev=0.01))
softmax_b_l = tf.Variable(tf.constant(0.1, shape=[num_classes]))

#ratio = (sum(train_y)/train_y.shape[0])
#class_weight = tf.constant([ratio[0], 1.0-ratio[0]])
logit_l = tf.matmul(last_l, softmax_w_l)+ softmax_b_l
prediction_l = tf.nn.softmax(logit_l)

# compute elementwise cross entropy.
loss_l = -tf.reduce_sum(target_l * tf.log(prediction_l))#class_weights
#f1w=tf.metrics.auc( tf.argmax(target, 1),  tf.argmax(prediction, 1) )
#mat=tf.confusion_matrix( tf.argmax(target, 1), tf.argmax(prediction, 1) ) 
#report=classification_report(  tf.argmax(target, 1), tf.argmax(prediction, 1)) 
# Loss function using L2 Regularization
regularizer_l = tf.nn.l2_loss(softmax_w_l)
cross_entropy_l = tf.reduce_mean(loss_l + beta_l * regularizer_l)

if(optim_l ==1):
    optimizer_l = tf.train.AdamOptimizer(learning_rate_l)
else:
    optimizer_l = tf.train.RMSPropOptimizer(learning_rate_l)

# Passing global_step to minimize() will increment it at each step.
#this has to be done here as it add additionals operations so have to be done befor init
minOpt_l = optimizer_l.minimize(cross_entropy_l, global_step=global_step)   #this throws warning of converting sparse to dense

mistakes_l = tf.not_equal(tf.argmax(target_l, 1), tf.argmax(prediction_l, 1))
testerror_l =  tf.reduce_mean(tf.cast(mistakes_l, tf.float32))


auc_l,aucl_op = tf.metrics.auc(tf.argmax(target_l, 1), tf.argmax(prediction_l, 1))
pr_l,prl_op = tf.metrics.precision(tf.argmax(target_l, 1), tf.argmax(prediction_l, 1))
rc_l,rcl_op = tf.metrics.recall(tf.argmax(target_l, 1), tf.argmax(prediction_l, 1))
f1_l = tf.divide(tf.scalar_mul(2,tf.multiply(pr_l,rc_l)),tf.add(pr_l, rc_l))
saver_l = tf.train.Saver()


N = train_X.shape[0]
max_iterations_l = int(np.floor(N/batch_size_l))

sess_l = tf.Session()
sess_l.run(tf.global_variables_initializer())
#to initialize optimizer
sess_l.run(tf.initialize_all_variables())
sess_l.run(tf.local_variables_initializer())

minima=100
#left
for epoch in range(epoch_count_l):
    epoch_error = 0
    permutation = list(np.random.permutation(N))
    shuffled_X = train_X[permutation,:,:]
    shuffled_y = train_y[permutation,:]
    for k in range(max_iterations_l): #iterations per epoch
        #ind_N = np.random.choice(N,batch_size,replace=False)
        dt_l = shuffled_X[k*batch_size_l:(k+1)*batch_size_l,:,left_rois]
        tgt_l =shuffled_y[k*batch_size_l:(k+1)*batch_size_l,:]
        out= sess_l.run([cross_entropy_l, minOpt_l], {
                    inputs_l: dt_l, target_l: tgt_l, dropout_l: dout_l})
        epoch_error += out[0]
    #for rest of the data
    dt_l = shuffled_X[(k+1)*batch_size_l:N,:,left_rois]
    tgt_l =shuffled_y[(k+1)*batch_size_l:N,:]
    out= sess_l.run([cross_entropy_l, minOpt_l], {
                    inputs_l: dt_l, target_l: tgt_l, dropout_l: dout_l})
    epoch_error += out[0]

    terror = sess_l.run(testerror_l, {
                    inputs_l: train_X[:,:,left_rois], target_l: train_y, dropout_l: 1})
    verror,y_predv = sess_l.run([testerror_l, prediction_l], {
                    inputs_l: valid_X[:,:,left_rois], target_l: valid_y, dropout_l: 1})
    f1l = sklearn.metrics.f1_score(tf.argmax(valid_y, 1), tf.argmax(y_predv, 1)) 
    prl = sklearn.metrics.precision_score(tf.argmax(valid_y, 1), tf.argmax(y_predv,1)) 
    rcl = sklearn.metrics.recall_score(tf.argmax(valid_y, 1), tf.argmax(y_predv,1)) 
    aucl = sklearn.metrics.roc_auc_score(tf.argmax(valid_y, 1), tf.argmax(y_predv,1)) 
    error,mis,y_predt = sess_l.run([testerror_l, mistakes_l,prediction_l], {
                    inputs_l: test_X[:,:,left_rois], target_l: test_y, dropout_l: 1})
    f1lt = sklearn.metrics.f1_score(tf.argmax(test_y, 1), tf.argmax(y_predt, 1)) 
    prlt = sklearn.metrics.precision_score(tf.argmax(test_y, 1), tf.argmax(y_predt,1)) 
    rclt = sklearn.metrics.recall_score(tf.argmax(test_y, 1), tf.argmax(y_predt,1)) 
    auclt = sklearn.metrics.roc_auc_score(tf.argmax(test_y, 1), tf.argmax(y_predt,1)) 
    lr_l = sess_l.run(learning_rate_l)
    if(optim_l == 1):
        print('L Adam ',seed_l, batch_size_l,dout_l, num_layers_l,hidden_size_l, epoch_count_l, beta_l, steps_l, starter_learning_rate_l, decay_rate_l  )
        str = "L Adam %d %d %.3f %d %d %d %.3f %d %.8f %.3f" %(seed_l, batch_size_l,dout_l, num_layers_l,hidden_size_l, epoch_count_l, beta_l, steps_l, lr_l, decay_rate_l  )
        logging.debug(str)
    else:
        print('L RMS ',seed_l, batch_size_l,dout_l, num_layers_l,hidden_size_l, epoch_count_l, beta_l, steps_l, starter_learning_rate_l, decay_rate_l  )
        str = "L RMS %d %d %.3f %d %d %d %.3f %d %.8f %.3f" %(seed_l, batch_size_l,dout_l, num_layers_l,hidden_size_l, epoch_count_l, beta_l, steps_l, lr_l, decay_rate_l  )
        logging.debug(str)

    if(verror < 0.2 and error < 0.2):
        logging.debug('left Saurabh  - possible solution *********************')
    if(verror < 0.25 and error < 0.25):
        logging.debug('left sgarg  - possible solution *********************')
        spio.savemat(misfile, {'mis': mis})
    
    if(verror < minima):
        minima = verror
        save_path = saver_l.save(sess_l,sessfile_l)
    print('L Epoch {:2d} cost {:3.1f} train_error {:3.1f}% valid_error {:3.1f}% test_error {:3.1f}%'.format(epoch + 1,epoch_error/max_iterations_l,terror*100,verror*100, 100 * error))
    logging.debug('L Epoch {:2d} cost {:3.1f} train_error {:3.1f}% valid_error {:3.1f}% test_error {:3.1f}%'.format(epoch + 1,epoch_error/max_iterations_l,terror*100,verror*100, 100 * error))
    logging.debug('L Valid AUC {:3.3f} PR{:3.3f} RC {:3.3f}% f1_score {:3.3f}%'.format(aucl,prl,rcl,f1l))
    logging.debug('L Test AUC {:3.3f} PR{:3.3f} RC {:3.3f}% f1_score {:3.3f}%'.format(auclt,prlt,rclt,f1lt))



# data = Batch size x time steps x right features
target_r= tf.placeholder(tf.float32, [None, num_classes])
inputs_r = tf.placeholder(tf.float32, [None, 90, 100])

with tf.variable_scope('right'):
    outputs_r, _ = tf.nn.dynamic_rnn(lstm_r, inputs_r, dtype=tf.float32)
output_r = tf.transpose(outputs_r, [1, 0, 2])
last_r = tf.gather(output_r, int(output_r.get_shape()[0]) - 1)
softmax_w_r = tf.Variable(tf.truncated_normal([hidden_size_r, num_classes], stddev=0.01))
softmax_b_r = tf.Variable(tf.constant(0.1, shape=[num_classes]))
logit_r = tf.matmul(last_r, softmax_w_r)+ softmax_b_r
prediction_r = tf.nn.softmax(logit_r)

loss_r = -tf.reduce_sum(target_r * tf.log(prediction_r))#class_weights
regularizer_r = tf.nn.l2_loss(softmax_w_r)
cross_entropy_r = tf.reduce_mean(loss_r + beta_r * regularizer_r)

if(optim_r ==1):
    optimizer_r = tf.train.AdamOptimizer(learning_rate_r)
else:
    optimizer_r = tf.train.RMSPropOptimizer(learning_rate_r)

minOpt_r = optimizer_r.minimize(cross_entropy_r, global_step=global_step)   #this throws warning of converting sparse to dense
mistakes_r = tf.not_equal(tf.argmax(target_r, 1), tf.argmax(prediction_r, 1))
testerror_r =  tf.reduce_mean(tf.cast(mistakes_r, tf.float32))

auc_r,aucr_op = tf.metrics.auc(tf.argmax(target_r, 1), tf.argmax(prediction_r, 1))
pr_r,prr_op = tf.metrics.precision(tf.argmax(target_r, 1), tf.argmax(prediction_r, 1))
rc_r,rcr_op = tf.metrics.recall(tf.argmax(target_r, 1), tf.argmax(prediction_r, 1))
#f1_r = 2*pr_r*rc_r/(pr_r + rc_r)
f1_r = tf.divide(tf.scalar_mul(2,tf.multiply(pr_r,rc_r)),tf.add(pr_r, rc_r))
saver_r = tf.train.Saver()

sess_r = tf.Session()
sess_r.run(tf.global_variables_initializer())
#to initialize optimizer
sess_r.run(tf.initialize_all_variables())
sess_r.run(tf.local_variables_initializer())


max_iterations_r = int(np.floor(N/batch_size_r))
#right
minima =100
for epoch in range(epoch_count_r):
    epoch_error = 0
    permutation = list(np.random.permutation(N))
    shuffled_X = train_X[permutation,:,:]
    shuffled_y = train_y[permutation,:]
    for k in range(max_iterations_r): #iterations per epoch
        #ind_N = np.random.choice(N,batch_size,replace=False)
        dt_r = shuffled_X[k*batch_size_r:(k+1)*batch_size_r,:,right_rois]
        tgt_r =shuffled_y[k*batch_size_r:(k+1)*batch_size_r,:]
        out= sess_r.run([cross_entropy_r, minOpt_r], {
                     inputs_r: dt_r, target_r: tgt_r, dropout_r: dout_r})
        epoch_error += out[0]
    #for rest of the data
    dt_r = shuffled_X[(k+1)*batch_size_r:N,:,right_rois]
    tgt_r =shuffled_y[(k+1)*batch_size_r:N,:]
    out= sess_r.run([cross_entropy_r, minOpt_r], {
                     inputs_r: dt_r, target_r: tgt_r, dropout_r: dout_r})
    epoch_error += out[0]

    terror = sess_r.run(testerror_r, {
                     inputs_r: train_X[:,:,right_rois], target_r: train_y, dropout_r: 1})
    verror,y_predr = sess_r.run([testerror_r, prediction_r], {
                     inputs_r: valid_X[:,:,right_rois], target_r: valid_y, dropout_r: 1})
    f1r = sklearn.metrics.f1_score(tf.argmax(valid_y, 1), tf.argmax(y_predr, 1)) 
    prr = sklearn.metrics.precision_score(tf.argmax(valid_y, 1), tf.argmax(y_predr,1)) 
    rcr = sklearn.metrics.recall_score(tf.argmax(valid_y, 1), tf.argmax(y_predr,1)) 
    aucr = sklearn.metrics.roc_auc_score(tf.argmax(valid_y, 1), tf.argmax(y_predr,1)) 
    error,mis,y_predrt = sess_r.run([testerror_r, mistakes_r, prediction_r], {
                     inputs_r: test_X[:,:,right_rois], target_r: test_y, dropout_r: 1})
    f1rt = sklearn.metrics.f1_score(tf.argmax(test_y, 1), tf.argmax(y_predrt, 1)) 
    prrt = sklearn.metrics.precision_score(tf.argmax(test_y, 1), tf.argmax(y_predrt,1)) 
    rcrt = sklearn.metrics.recall_score(tf.argmax(test_y, 1), tf.argmax(y_predrt,1)) 
    aucrt = sklearn.metrics.roc_auc_score(tf.argmax(test_y, 1), tf.argmax(y_predrt,1)) 
    lr_r = sess_r.run(learning_rate_r)
    if(optim_r == 1):
        print('R Adam ',seed_r, batch_size_r,dout_r, num_layers_r,hidden_size_r, epoch_count_r, beta_r, steps_r, lr_r, decay_rate_r  )
        str = "R Adam %d %d %.3f %d %d %d %.3f %d %.8f %.3f" %(seed_r, batch_size_r,dout_r, num_layers_r,hidden_size_r, epoch_count_r, beta_r, steps_r, lr_r, decay_rate_r  )
        logging.debug(str)
    else:
        print('R RMS ',seed_r, batch_size_r,dout_r, num_layers_r,hidden_size_r, epoch_count_r, beta_r, steps_r, lr_r, decay_rate_r  )
        str = "R RMS  %d %d %.3f %d %d %d %.3f %d %.8f %.3f" %(seed_r, batch_size_r,dout_r, num_layers_r,hidden_size_r, epoch_count_r, beta_r, steps_r, lr_r, decay_rate_r  )
        logging.debug(str)

    if(verror < 0.2 and error < 0.2):
        logging.debug('right Saurabh  - possible solution *********************')
    if(verror < 0.25 and error < 0.25):
        logging.debug('right sgarg  - possible solution *********************')
        spio.savemat(misfile, {'mis': mis})
    if(verror < minima):
        minima = verror
        save_path = saver_r.save(sess_r,sessfile_r)
    print('right Epoch {:2d} cost {:3.1f} train_error {:3.1f}% valid_error {:3.1f}% test_error {:3.1f}%'.format(epoch + 1,epoch_error/max_iterations_r,terror*100,verror*100, 100 * error))
    logging.debug('right Epoch {:2d} cost {:3.1f} train_error {:3.1f}% valid_error {:3.1f}% test_error {:3.1f}%'.format(epoch + 1,epoch_error/max_iterations_r,terror*100,verror*100, 100 * error))
    logging.debug('R Valid AUC {:3.3f} PR{:3.3f} RC {:3.3f}% f1_score {:3.3f}%'.format(aucr,prr,rcr,f1r))
    logging.debug('R Test AUC {:3.3f} PR{:3.3f} RC {:3.3f}% f1_score {:3.3f}%'.format(aucrt,prrt,rcrt,f1rt))


#TODO
softmax_w = tf.Variable(tf.truncated_normal([hidden_size_l + hidden_size_r, num_classes], stddev=0.01))
softmax_b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
learning_rate = 0.00001 
inputs = tf.placeholder(tf.float32, [None, hidden_size_l+hidden_size_r])
target= tf.placeholder(tf.float32, [None, num_classes])
logit = tf.matmul(inputs, softmax_w)+ softmax_b
prediction = tf.nn.softmax(logit)
loss = -tf.reduce_sum(target * tf.log(prediction))#class_weights
cross_entropy = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate)
minOpt = optimizer.minimize(cross_entropy, global_step=global_step)   #this throws warning of converting sparse to dense
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
testerror =  tf.reduce_mean(tf.cast(mistakes, tf.float32))

auc,auc_op = tf.metrics.auc(tf.argmax(target, 1), tf.argmax(prediction, 1))
pr,pr_op = tf.metrics.precision(tf.argmax(target, 1), tf.argmax(prediction, 1))
rc,rc_op = tf.metrics.recall(tf.argmax(target, 1), tf.argmax(prediction, 1))
#f1 = 2*pr*rc/(pr + rc)
f1 = tf.divide(tf.scalar_mul(2,tf.multiply(pr,rc)),tf.add(pr, rc))

saver_l.restore(sess_l, sessfile_l)
saver_r.restore(sess_r, sessfile_r)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#to initialize optimizer
sess.run(tf.initialize_all_variables())
sess.run(tf.local_variables_initializer())
for epoch in range(epoch_count_l):
    epoch_error = 0
    permutation = list(np.random.permutation(N))
    shuffled_X = train_X[permutation,:,:]
    shuffled_y = train_y[permutation,:]
    for k in range(max_iterations_l): #iterations per epoch
        #ind_N = np.random.choice(N,batch_size,replace=False)
        dtl = shuffled_X[(k)*batch_size_l:(k+1)*batch_size_l,:,left_rois]
        dtr = shuffled_X[(k)*batch_size_l:(k+1)*batch_size_l,:,right_rois]
        tgt =shuffled_y[k*batch_size_l:(k+1)*batch_size_l,:]
        inpl = sess_l.run(last_l, feed_dict={
                    inputs_l: dtl, target_l: tgt, dropout_l: 1})
        inpr = sess_r.run(last_r, feed_dict={
                    inputs_r: dtr, target_r: tgt, dropout_r: 1})

        dt = sess.run(tf.concat([inpl, inpr],1))
        out= sess.run([cross_entropy, minOpt], {
                    inputs: dt, target: tgt})
        epoch_error += out[0]
    #for rest of the data
    dtl = shuffled_X[(k+1)*batch_size_l:N,:,left_rois]
    dtr = shuffled_X[(k+1)*batch_size_l:N,:,right_rois]
    tgt =shuffled_y[(k+1)*batch_size_l:N,:]

    inpl = sess_l.run(last_l, feed_dict={inputs_l: dtl, target_l: tgt, dropout_l: 1})
    inpr = sess_r.run(last_r, feed_dict={inputs_r: dtr, target_r: tgt, dropout_r: 1})
    dt = sess.run(tf.concat([inpl, inpr],1))
    out= sess.run([cross_entropy, minOpt], {inputs: dt, target: tgt})

    epoch_error += out[0]

    inpl = sess_l.run(last_l, {
        inputs_l: train_X[:,:,left_rois], target_l: train_y, dropout_l: 1})
    inpr = sess_r.run(last_r, {
        inputs_r: train_X[:,:,right_rois], target_r: train_y, dropout_r: 1})
    dt = sess.run(tf.concat([inpl, inpr],1))
    terror= sess.run(testerror, {inputs: dt, target: train_y})
    
    inpl = sess_l.run(last_l, {
        inputs_l: valid_X[:,:,left_rois], target_l: valid_y, dropout_l: 1})
    inpr = sess_r.run(last_r, {
        inputs_r: valid_X[:,:,right_rois], target_r: valid_y, dropout_r: 1})
    dt = sess.run(tf.concat([inpl, inpr],1))
    verror,y_preda= sess.run([testerror,prediction], {inputs: dt, target: valid_y})
    f1a = sklearn.metrics.f1_score(tf.argmax(valid_y, 1), tf.argmax(y_preda, 1)) 
    pra = sklearn.metrics.precision_score(tf.argmax(valid_y, 1), tf.argmax(y_preda,1)) 
    rca = sklearn.metrics.recall_score(tf.argmax(valid_y, 1), tf.argmax(y_preda,1)) 
    auca = sklearn.metrics.roc_auc_score(tf.argmax(valid_y, 1), tf.argmax(y_preda,1)) 
    
    inpl = sess_l.run(last_l, {
        inputs_l: test_X[:,:,left_rois], target_l: test_y, dropout_l: 1})
    inpr = sess_r.run(last_r, {
        inputs_r: test_X[:,:,right_rois], target_r: test_y, dropout_r: 1})
    dt = sess.run(tf.concat([inpl, inpr],1))
    error,mis,y_predat = sess.run([testerror, mistakes,prediction], {
                    inputs: dt, target: test_y})
    f1at = sklearn.metrics.f1_score(tf.argmax(test_y, 1), tf.argmax(y_predat, 1)) 
    prat = sklearn.metrics.precision_score(tf.argmax(test_y, 1), tf.argmax(y_predat,1)) 
    rcat = sklearn.metrics.recall_score(tf.argmax(test_y, 1), tf.argmax(y_predat,1)) 
    aucat = sklearn.metrics.roc_auc_score(tf.argmax(test_y, 1), tf.argmax(y_predat,1)) 

    if(verror < 0.2 and error < 0.2):
        logging.debug('all Saurabh  - possible solution *********************')
    if(verror < 0.25 and error < 0.25):
        logging.debug('all sgarg  - possible solution *********************')
        spio.savemat(misfile, {'mis': mis})
    
    print('A Epoch {:2d} cost {:3.1f} train_error {:3.1f}% valid_error {:3.1f}% test_error {:3.1f}%'.format(epoch + 1,epoch_error/max_iterations_l,terror*100,verror*100, 100 * error))
    logging.debug('A Epoch {:2d} cost {:3.1f} train_error {:3.1f}% valid_error {:3.1f}% test_error {:3.1f}%'.format(epoch + 1,epoch_error/max_iterations_l,terror*100,verror*100, 100 * error))
    logging.debug('A Valid AUC {:3.3f} PR{:3.3f} RC {:3.3f}% f1_score {:3.3f}%'.format(auca,pra,rca,f1a))
    logging.debug('A Test AUC {:3.3f} PR{:3.3f} RC {:3.3f}% f1_score {:3.3f}%'.format(aucat,prat,rcat,f1at))


