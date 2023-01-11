import argparse

import torch
from torch import nn, optim
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
from VAE import VAE
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_train_samples", default=-1, type=int)
parser.add_argument("-s", "--save_name", default="result.pkl", type=str)
parser.add_argument("-p", "--parent", default=5, type=int)
parser.add_argument("-v", "--variates", default=None, type=int)
parser.add_argument("--show", action="store_true")
parser.add_argument("--latent", default=5, type=int)
parser.add_argument("--gpu",action="store_true")
parser.add_argument("-r","--learning_rate",default=0.001,type=float)
parser.add_argument("-e","--epoch",default=80,type=int)
parser.add_argument("-b","--batch",default=1024,type=int)

args = parser.parse_args()

num_train_samples = args.num_train_samples
save_name = args.save_name
parent = args.parent
variates = args.variates
show = args.show
latent = args.latent
gpu=args.gpu
learning_rate=args.learning_rate
epoch=args.epoch
batch_size=args.batch

trainset_filename = "ServerMachineDataset/train/pkl/machine-1-1.pkl"
testset_filename = "ServerMachineDataset/test/pkl/machine-1-1.pkl"
testset_gt_filename = "ServerMachineDataset/test_label/pkl/machine-1-1.pkl"
dataloader = DataLoader(trainset_filename, testset_filename, testset_gt_filename, n_variate=variates)
X = dataloader.load_causal_data()
# Record = ges(X, maxP=5)
with open("save/machine.1.1.pkl", "rb") as f:
    #     pickle.dump(Record,f)
    Record = pickle.load(f)
# print(Record['G'].graph)
# print(Record['G'])
dataloader.prepare_ad_data(Record['G'].graph, univariate=True)
R_trainset_x, P_trainset_x = dataloader.load_train_data()
R_testset_x, P_testset_x = dataloader.load_test_data()

train_set = torch.Tensor(P_trainset_x[0])
test_set = torch.Tensor(P_testset_x[0])
print(train_set.shape)
print(test_set.shape)

input_size = train_set.shape[1]
latent_size = args.latent
train_set_size=train_set.shape[0]
print(input_size)

model = VAE(input_size,latent_size)
if gpu:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse_loss=nn.MSELoss()

for epochs in range(epoch):
    if epochs % 10 == 0:
        permutation = np.random.permutation(train_set.shape[0])
        train_set = train_set[permutation]

    iter = train_set_size // batch_size
    with tqdm(total=iter, ascii=True) as pbar:
        pbar.set_postfix_str("epochs: --- train loss: -.------ test loss: -.------")
        for i in range(iter):
            batch_x = train_set[i * batch_size:(i + 1) * batch_size]
            if gpu:
                device = "cuda:" + str(gpu)
                batch_x = batch_x.cuda()
            recon, mu, log_std = model(batch_x)
            optimizer.zero_grad()
            loss = model.loss_function(recon, batch_x, mu, log_std)
            recon_loss=mse_loss(recon,batch_x)
            pbar.set_postfix_str(
                "epochs: %d/%d train loss: %.6f test loss: -.------" % (epochs + 1, epoch, loss.item()))
            loss.backward()
            optimizer.step()
            pbar.update()

        if iter * batch_size != train_set_size:
            batch_x = train_set[iter * batch_size:]
            if gpu:
                batch_x = batch_x.cuda()
            recon, mu, log_std = model(batch_x)
            optimizer.zero_grad()
            loss = model.loss_function(recon, batch_x, mu, log_std)
            loss.backward()
            optimizer.step()
            # pbar.set_postfix_str("epochs: %d train loss: %.6f test loss: -.------" % (epochs, loss.item()))
# exit()


# for i in dataloader.parent_list:
#     if len(i)!=0:
#         print(len(i)+1)
#
# print(dataloader.parent_list)
#
# print(np.shape(R_trainset_x))
# for i in P_trainset_x:
#     print(np.shape(i))
#
# print(np.shape(R_testset_x))
# for i in P_testset_x:
#     print(np.shape(i))

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

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# (2, 28457, 20)
# (28478, 7) 7
# (28478, 7) 7
# (28478, 6) 6
# (28478, 7) 7
# (28478, 4) 4
# (28478, 2) 2
# (28478, 7) 7
# (28478, 6) 1
# (28478, 7) 6
# (28478, 7) 7
# (28478, 7) 7
# (28478, 7) 7
# (28478, 2) 7
# (28478, 7) 2
# (28478, 7) 7
# (28478, 3) 7
# (28478, 7) 3
# (28478, 6) 7
# (28478, 7) 6
# (28478, 7) 7
# (28478, 3) 7
# (28478, 4) 3
# (28478, 7) 4
# (28478, 2) 7
# (28478, 7) 2
# (28478, 6) 7
# (28478, 4) 6
