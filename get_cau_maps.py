import os.path
import pickle

import numpy as np
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.ScoreBased.GES import ges
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--filename", type=str)
args = parser.parse_args()

# f=open("save/machine.1.1.pkl","rb")
# rec=pickle.load(f)
# f.close()
# print(rec['G'])
# exit()

'''machine-1-1.camap.pkl
machine-1-2.camap.pkl
machine-1-3.camap.pkl
machine-1-4.camap.pkl
machine-1-5.camap.pkl
machine-1-6.camap.pkl
machine-1-7.camap.pkl
machine-1-8.camap.pkl
machine-2-1.camap.pkl
machine-2-2.camap.pkl
machine-2-3.camap.pkl
machine-2-4.camap.pkl
machine-2-5.camap.pkl
machine-2-6.camap.pkl
machine-2-7.camap.pkl
machine-2-8.camap.pkl
machine-2-9.camap.pkl
machine-3-1.camap.pkl
machine-3-2.camap.pkl
machine-3-3.camap.pkl
machine-3-4.camap.pkl
machine-3-5.camap.pkl
machine-3-6.camap.pkl
machine-3-7.camap.pkl
machine-3-8.camap.pkl
machine-3-9.camap.pkl
machine-3-10.camap.pkl
machine-3-11.camap.pkl'''


pkl_path='ServerMachineDataset/train/pkl'
file_name=args.filename
#
# for root,path_list,files in os.walk("ServerMachineDataset/train/pkl"):
#     for file in files:
f=open(os.path.join(pkl_path,file_name),"rb")
train_data=pickle.load(f)
f.close()

non_zero_variate = np.where(np.sum(train_data, axis=0) != 0)[0]
non_zero_data_train = train_data.transpose()[non_zero_variate].transpose()
print(non_zero_data_train.shape)

X=non_zero_data_train
Record = ges(X, maxP=5)

f=open(os.path.join('save/maps',file_name[:-3]+'camap.pkl'),'wb')
pickle.dump(Record,f)
f.close()


exit()
f= open(os.path.join(pkl_path,file_name),"rb")
train_data=pickle.load(f)
f.close()

non_zero_variate = np.where(np.sum(train_data, axis=0) != 0)[0]
print(np.sum(train_data,axis=0))
print(non_zero_variate)
non_zero_data_train = train_data.transpose()[non_zero_variate].transpose()
print(non_zero_data_train.shape)

X=non_zero_data_train[:,:10]
Record = ges(X, maxP=5)

f=open(os.path.join('save/maps',file_name[:-3]+'camap.pkl'),'wb')
pickle.dump(Record,f)
f.close()

