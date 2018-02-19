#keeps the data in 3D 
from os import listdir
import numpy as np
import scipy.io as spio
import scipy.stats as spstats
from sklearn.preprocessing import MinMaxScaler

filelist = listdir('/home/saurabh/scratch/data')
data=np.empty(0)
wid =90
jump = 60
tlabels=[]
slabels=[]
mlabels=[]
sid=[]
count=0;
scaler = MinMaxScaler(feature_range=(0, 1))
for f in filelist:
    file = '/home/saurabh/scratch/data/'+f 
    dt = spio.loadmat(file)
    d1 = dt['a']
    count=count+1
    #if resting state or not
    if (d1.shape[1]>=230 and d1.shape[1] <= 240):
        resting=1
    else:
        resting=0

    st = f.split('_')
    for i in range(len(st)):
        id = st[i]
        if(id[0] == 'P' or id[0] == 'H' or id[0]=='N' or id[0]=='T' or id[0]=='F'):
            if(id[0] == 'P' or id[0]=='T' or id[0]=='F'):
                subjlabel= 1
                jump=60
            else:
                subjlabel=0
                jump=10
            if(id[0] == 'P' and id[-1]=='A'):
                onmed =0    # if offmed then reject that label
            else:
                onmed=1
        if(id[0]=='T' and id[1]=='E'): #remove TMS data
                onmed=0
            

    print(f, subjlabel, onmed)
    for x in range(0,d1.shape[1],jump):
        if(x+wid < d1.shape[1]):
            #data is empty
            if(not data.any()):
                sid.append(count)
                #perform zscore normalization
                tempdata2 =spstats.zscore(d1[:,x:x+wid],1)
                tempdata1 = scaler.fit_transform(tempdata2)
                tempdata = np.atleast_3d(tempdata2.T).T
                data = tempdata
                tlabels.append(resting) 
                slabels.append(subjlabel) 
                mlabels.append(onmed) 
            else:
                sid.append(count)
                tempdata2 =spstats.zscore(d1[:,x:x+wid],1)
                tempdata1 = scaler.fit_transform(tempdata2)
                tempdata = np.atleast_3d(tempdata2.T).T
                data = np.asarray(data)   
                data = np.vstack((data,tempdata))
                tlabels.append(resting) 
                slabels.append(subjlabel) 
                mlabels.append(onmed) 
        if(d1.shape[1]-wid > x + jump/2): 
            sid.append(count)
            tempdata2 = spstats.zscore(d1[:,d1.shape[1]-wid:d1.shape[1]],1)
            tempdata1 = scaler.fit_transform(tempdata2)
            tempdata = np.atleast_3d(tempdata2.T).T
            data = np.asarray(data)   
            data = np.vstack((data,tempdata))
            tlabels.append(resting) 
            slabels.append(subjlabel)
            mlabels.append(onmed)
spio.savemat('/home/saurabh/scratch/saved_data/loaddata_rest90d_wo_minmax.mat', {'a': data})
spio.savemat('/home/saurabh/scratch/saved_data/slabels_rest90d_wo_minmax.mat', {'s': slabels}) #pd or HC
spio.savemat('/home/saurabh/scratch/saved_data/tlabels_rest90d_wo_minmax.mat', {'t': tlabels}) #restomg
spio.savemat('/home/saurabh/scratch/saved_data/mlabels_rest90d_wo_minmax.mat', {'m': mlabels}) #onmed or offmed
spio.savemat('/home/saurabh/scratch/saved_data/sid90d_wo_minmax.mat', {'m': sid}) #onmed or offmed
