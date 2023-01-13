import os.path
import pickle

import numpy as np
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.ScoreBased.GES import ges
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# f=open("save/machine.1.1.pkl","rb")
# rec=pickle.load(f)
# f.close()
# print(rec['G'])
# exit()

pkl_path='ServerMachineDataset/train/pkl'
file_name="machine-1-1.pkl"

for root,path_list,files in os.walk("ServerMachineDataset/train/pkl"):
    for file in files:
        f=open(os.path.join(root,file),"rb")
        train_data=pickle.load(f)
        f.close()

        non_zero_variate = np.where(np.sum(train_data, axis=0) != 0)[0]
        non_zero_data_train = train_data.transpose()[non_zero_variate].transpose()
        print(non_zero_data_train.shape)

        X=non_zero_data_train
        Record = ges(X, maxP=5)

        f=open(os.path.join('save/maps',file[:-3]+'camap.pkl'),'wb')
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

