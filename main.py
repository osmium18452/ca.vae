import argparse
from causallearn.search.ScoreBased.GES import ges

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
import pickle
import os
import numpy as np
from DataLoader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_train_samples", default=-1, type=int)
parser.add_argument("-s", "--save_name", default="result.pkl", type=str)
parser.add_argument("-p", "--parent", default=5, type=int)
parser.add_argument("-v", "--variates", default=-1, type=int)
parser.add_argument("--show", action="store_true")

args = parser.parse_args()

num_train_samples=args.num_train_samples
save_name=args.save_name
parent=args.parent
variates=args.variates
show=args.show

trainset_filename = "ServerMachineDataset/train/pkl/machine-1-1.pkl"
testset_filename = "ServerMachineDataset/test/pkl/machine-1-1.pkl"
testset_gt_filename = "ServerMachineDataset/test_label/pkl/machine-1-1.pkl"
dataloader = DataLoader(trainset_filename, testset_filename, testset_gt_filename, n_variate=15)
X = dataloader.load_causal_data()
# Record = ges(X, maxP=5)
with open("save/testgraph.pkl", "rb") as f:
    #     pickle.dump(Record,f)
    Record = pickle.load(f)
# print(Record['G'].graph)
# print(Record['G'])
dataloader.prepare_ad_data(Record['G'].graph, univariate=True)
(R_trainset_x, R_trainset_y), (P_trainset_x, P_trainset_y)=dataloader.load_train_data()
(R_testset_x, R_testset_y), (P_testset_x, P_testset_y)=dataloader.load_test_data()

print(np.shape(R_trainset_x))
print(np.shape(R_trainset_y))
print(np.shape(P_trainset_y))
for i in P_trainset_x:
    print(np.shape(i))

print(np.shape(R_testset_x))
print(np.shape(R_testset_y))
print(np.shape(P_testset_y))
for i in P_testset_x:
    print(np.shape(i))
# print(dataloader.non_zero_data.shape)
# print(dataloader.load_non_zero_variate())
# pyd = GraphUtils.to_pydot(Record['G'],labels=list(range(dataloader.load_num_non_zero_variate())))
# tmp_png = pyd.create_png(f="png")
# fp = io.BytesIO(tmp_png)
# img = mpimg.imread(fp, format='png')
# plt.axis('off')
# plt.imshow(img)
# plt.show()

# train_data_path="ServerMachineDataset/train/pkl"
# train_data_file_name="machine-1-1.pkl"
# f=open(os.path.join(train_data_path,train_data_file_name),"rb")
# data=pickle.load(f)
# f.close()
# print(data.shape)
#
# zero_line=np.where(np.sum(data,axis=0)==0)
# non_zero_line=np.where(np.sum(data,axis=0)!=0)
# non_zero_data=data.transpose()[non_zero_line].transpose()
# print(zero_line)
# print(non_zero_line)
# print(non_zero_data.shape)
# # exit()
#
# X=non_zero_data[:num_train_samples,:variates]
# print("getting graph")
# Record = ges(X, maxP=parent)
# print("finished")
# if not os.path.exists("save"):
#     os.makedirs("save")
# f= open(os.path.join("save",save_name),"wb")
# pickle.dump(Record,f)
# f.close()
#
# if show:
#     pyd = GraphUtils.to_pydot(Record['G'])
#     tmp_png = pyd.create_png(f="png")
#     fp = io.BytesIO(tmp_png)
#     img = mpimg.imread(fp, format='png')
#     plt.axis('off')
#     plt.imshow(img)
#     plt.show()

#
# # or save the graph
# pyd.write_png('simple_test.png')