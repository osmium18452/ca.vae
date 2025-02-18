import argparse

import numpy
import torch
from torch import nn, optim
import torch.nn.functional as F
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
from InfernocusVAE import InfernocusVAE
from VAE import VAE
from AE import AE
from CNN import CNN
from tqdm import tqdm

# with open("ServerMachineDataset/test/pkl/machine-1-1.pkl","rb") as f:
#     data=pickle.load(f)
# print(data.shape)
# exit()

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_train_samples", default=-1, type=int)
parser.add_argument("-s", "--save_name", default="result.pkl", type=str)
parser.add_argument("-p", "--parent", default=5, type=int)
parser.add_argument("-v", "--variates", default=None, type=int)
parser.add_argument("--show", action="store_true")
parser.add_argument("--latent", default=5, type=int)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("-r", "--learning_rate", default=0.001, type=float)
parser.add_argument("-e", "--epoch", default=10, type=int)
parser.add_argument("-b", "--batch", default=1024, type=int)
parser.add_argument("-m", "--multivariate", action="store_true")
parser.add_argument("-w", "--window_size", default=20, type=int)
parser.add_argument("-g", "--gpu_device", default="0", type=str)
parser.add_argument('--cnn_lr', default=0.0001, type=float)
parser.add_argument('--vae_lr', default=0.001, type=float)
parser.add_argument('-N', '--normalize_data', action='store_true')
args = parser.parse_args()

num_train_samples = args.num_train_samples
save_name = args.save_name
parent = args.parent
variates = args.variates
show = args.show
latent = args.latent
gpu = args.gpu
learning_rate = args.learning_rate
epoch = args.epoch
batch_size = args.batch
univariate = not args.multivariate
multivariate = args.multivariate
window_size = args.window_size
gpu_device = args.gpu_device
cnn_lr = args.cnn_lr
vae_lr = args.vae_lr
normalize_data = args.normalize_data

if gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device

trainset_filename = "ServerMachineDataset/train/pkl/machine-2-1.pkl"
testset_filename = "ServerMachineDataset/test/pkl/machine-2-1.pkl"
testset_gt_filename = "ServerMachineDataset/test_label/pkl/machine-2-1.pkl"
dataloader = DataLoader(trainset_filename, testset_filename, testset_gt_filename, n_variate=variates,
                        data_normalization=normalize_data)
X = dataloader.load_causal_data()
# print(X.shape)
# exit()
# Record = ges(X, maxP=5)
with open("save/machine-2-1.camap.pkl", "rb") as f:
    #     pickle.dump(Record,f)
    Record = pickle.load(f)
# print(Record['G'].graph)
# print(Record['G'])
dataloader.prepare_ad_data(Record['G'].graph, univariate=univariate, window_size=window_size)
# if univariate:
#     R_trainset_x, R_trainset_y, P_trainset_x = dataloader.load_train_data(univariate=univariate)
#     R_testset_x, R_testset_y, P_testset_x = dataloader.load_test_data(univariate=univariate)
# else:
#     R_trainset_x, P_trainset_x = dataloader.load_train_data(univariate=univariate)
#     R_testset_x, P_testset_x = dataloader.load_test_data(univariate=univariate)

# train infernocusVAE
'''infernocus_input_list=dataloader.load_P_input_len_list()
infernocus_train_data_set = torch.Tensor(dataloader.load_infernocus_train_data_P())
train_set_size=infernocus_train_data_set.shape[0]
# print(infernocus_input_list,len(infernocus_input_list))
ivae = InfernocusVAE(infernocus_input_list, latent)
print(ivae.parameters() )
if gpu:
    ivae.cuda()
optimizer=optim.Adam(ivae.parameters(),lr=learning_rate)
mse_loss = nn.MSELoss()

for epochs in range(epoch):
    if epochs % 10 == 0:
        permutation = np.random.permutation(infernocus_train_data_set.shape[0])
        infernocus_train_data_set = infernocus_train_data_set[permutation]

    iter = train_set_size // batch_size
    with tqdm(total=iter, ascii=True) as pbar:
        pbar.set_postfix_str("epochs: --- train loss: -.------ test loss: -.------")
        for i in range(iter):
            batch_x = infernocus_train_data_set[i * batch_size:(i + 1) * batch_size]
            if gpu:
                device = "cuda:" + str(gpu)
                batch_x = batch_x.cuda()
            recon, mu, log_std = ivae(batch_x)
            optimizer.zero_grad()
            loss = ivae.loss_function(recon, batch_x, mu, log_std)
            recon_loss = mse_loss(recon, batch_x)
            pbar.set_postfix_str(
                "epochs: %d/%d train loss: %.6f test loss: -.------" % (epochs + 1, epoch, recon_loss.item()))
            loss.backward()
            optimizer.step()
            pbar.update()

        if iter * batch_size != train_set_size:
            batch_x = infernocus_train_data_set[iter * batch_size:]
            if gpu:
                batch_x = batch_x.cuda()
            recon, mu, log_std = ivae(batch_x)
            optimizer.zero_grad()
            loss = ivae.loss_function(recon, batch_x, mu, log_std)
            loss.backward()
            optimizer.step()

# recon,mu,log_std= ivae(infernocus_train_data_set)
# loss=ivae.loss_function(recon, infernocus_train_data_set, mu, log_std)
# print("xi",xi_recon)
# print(recon.shape,mu.shape,log_std.shape)
# print(np.sum(infernocus_input_list))
# print("loss",loss.shape)
# exit()'''

print("""train a lot of cnns""")
train_set_x, train_set_y = dataloader.load_cnn_train_data()
train_set_x = torch.Tensor(train_set_x).transpose(-1, -2)
train_set_y = torch.Tensor(train_set_y)
print(train_set_x.shape, train_set_y.shape)
train_set_size = train_set_x.shape[1]
cnn_num = train_set_x.shape[0]
print(train_set_size, cnn_num)

cnn_list = []
cnn_optimizer_list = []
for i in range(cnn_num):
    cnn_list.append(CNN(window_size))
    if gpu:
        cnn_list[-1].cuda()
    cnn_optimizer_list.append(optim.Adam(cnn_list[-1].parameters(), lr=cnn_lr))

mse_loss = nn.MSELoss()

for epochs in range(epoch):
    if epochs % 10 == 0:
        permutation = np.random.permutation(train_set_x.shape[1])
        train_set_x = train_set_x[:, permutation]
        train_set_y = train_set_y[:, permutation]
        # print(train_set_x.shape,train_set_y.shape)
    # exit()

    iter = train_set_size // batch_size
    with tqdm(total=iter, ascii=True) as pbar:
        pbar.set_postfix_str("epochs: --- train loss: -.-----e--- mse loss: -.-----e---")
        for i in range(iter):
            big_batch_x = train_set_x[:, i * batch_size:(i + 1) * batch_size]
            big_batch_y = train_set_y[:, i * batch_size:(i + 1) * batch_size]
            mse = 0.
            for j in range(cnn_num):
                batch_x = big_batch_x[j]
                batch_y = big_batch_y[j]
                # print(batch_x.shape,batch_y.shape)
                if gpu:
                    device = "cuda:" + str(gpu)
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                cnn_list[j].train()
                recon = cnn_list[j](batch_x)
                # print("recon",recon.shape)
                # exit()
                cnn_optimizer_list[j].zero_grad()
                loss = cnn_list[j].loss_function(recon, batch_y)
                # print(loss)
                mse += loss
                loss.backward()
                cnn_optimizer_list[j].step()
            mse /= cnn_num
            pbar.set_postfix_str(
                "epochs: %d/%d train loss: %.5e mse loss: %.5e" % (epochs + 1, epoch, loss.item(), mse))
            pbar.update()

        if iter * batch_size != train_set_size:
            big_batch_x = train_set_x[:, iter * batch_size:]
            big_batch_y = train_set_y[:, iter * batch_size:]
            for j in range(cnn_num):
                batch_x = big_batch_x[j]
                batch_y = big_batch_y[j]
                if gpu:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                cnn_list[j].train()
                recon = cnn_list[j](batch_x)
                cnn_optimizer_list[j].zero_grad()
                loss = cnn_list[j].loss_function(recon, batch_y)
                loss.backward()
                cnn_optimizer_list[j].step()

# exit()

print("""train a lot of vaes""")
train_set = torch.Tensor(dataloader.load_infernocus_train_data_P())
train_set_size = train_set.shape[0]
input_size_list = dataloader.load_P_input_len_list()
slice_list = [0, ]
for i in input_size_list:
    slice_list.append(slice_list[-1] + i)
vae_num = len(input_size_list)

vae_list = []
optimizer_list = []
for i in range(vae_num):
    vae_list.append(VAE(input_size=input_size_list[i], latent_size=latent))
    if gpu:
        vae_list[-1].cuda()
    optimizer_list.append(optim.Adam(vae_list[-1].parameters(), lr=vae_lr))

mse_loss = nn.MSELoss()

for epochs in range(epoch):
    if epochs % 10 == 0:
        permutation = np.random.permutation(train_set.shape[0])
        train_set = train_set[permutation]

    iter = train_set_size // batch_size
    with tqdm(total=iter, ascii=True) as pbar:
        pbar.set_postfix_str("epochs: --- train loss: -.-----e--- mse loss: -.-----e---")
        for i in range(iter):
            big_batch_x = train_set[i * batch_size:(i + 1) * batch_size]
            mse = 0.
            for j in range(vae_num):
                batch_x = big_batch_x[:, slice_list[j]:slice_list[j + 1]]
                if gpu:
                    device = "cuda:" + str(gpu)
                    batch_x = batch_x.cuda()
                recon, mu, log_std = vae_list[j](batch_x)
                optimizer_list[j].zero_grad()
                loss = vae_list[j].loss_function(recon, batch_x, mu, log_std)
                mse += F.mse_loss(recon, batch_x).item()
                loss.backward()
                optimizer_list[j].step()
            mse /= vae_num
            pbar.set_postfix_str(
                "epochs: %d/%d train loss: %.5e mse loss: %.5e" % (epochs + 1, epoch, loss.item(), mse))
            pbar.update()

        if iter * batch_size != train_set_size:
            big_batch_x = train_set[iter * batch_size:]
            for j in range(vae_num):
                batch_x = big_batch_x[:, slice_list[j]:slice_list[j + 1]]
                if gpu:
                    batch_x = batch_x.cuda()
                recon, mu, log_std = vae_list[j](batch_x)
                optimizer_list[j].zero_grad()
                loss = vae_list[j].loss_function(recon, batch_x, mu, log_std)
                loss.backward()
                optimizer_list[j].step()

# test phase cnn

test_set_x, test_set_y = dataloader.load_cnn_test_data()
cnn_ground_truth = test_set_y.squeeze().transpose()
print("test set", test_set_x.shape)
test_set_x = torch.Tensor(test_set_x).transpose(-1, -2)
test_set_size = test_set_y.shape[1]
cnn_recon_list = [[] for i in range(cnn_num)]
iter = test_set_size // batch_size
with tqdm(total=iter, ascii=True) as pbar:
    for i in range(iter):
        big_batch_x = test_set_x[:, i * batch_size:(i + 1) * batch_size]
        for j in range(cnn_num):
            batch_x = big_batch_x[j]
            if gpu:
                device = "cuda:" + str(gpu)
                batch_x = batch_x.cuda()
            cnn_list[j].eval()
            recon = cnn_list[j](batch_x)
            recon = recon.cpu()
            cnn_recon_list[j].append(recon.detach().numpy())
        pbar.update()

    if iter * batch_size != test_set_size:
        big_batch_x = test_set_x[:, iter * batch_size:]
        for j in range(cnn_num):
            batch_x = big_batch_x[j]
            if gpu:
                batch_x = batch_x.cuda()
            cnn_list[j].eval()
            recon = cnn_list[j](batch_x)
            recon = recon.cpu()
            cnn_recon_list[j].append(recon.detach().numpy())

# test phase vae
test_set_x = torch.Tensor(dataloader.load_infernocus_test_data_P())
print('vae test set', test_set_x.shape)
vae_ground_truth = test_set_x[:, slice_list[:-1]].numpy()[window_size + 1:]
test_set_size = test_set_x.shape[0]
vae_recon_list = [[] for i in range(vae_num)]
test_set_size = test_set_x.shape[0]
iter = test_set_size // batch_size
with tqdm(total=iter, ascii=True) as pbar:
    for i in range(iter):
        big_batch_x = test_set_x[i * batch_size:(i + 1) * batch_size]
        for j in range(vae_num):
            batch_x = big_batch_x[:, slice_list[j]:slice_list[j + 1]]
            if gpu:
                device = "cuda:" + str(gpu)
                batch_x = batch_x.cuda()
            recon, mu, log_std = vae_list[j](batch_x)
            recon = recon.cpu()
            # print(recon.shape)
            vae_recon_list[j].append(recon.detach().numpy()[:, 0])
        pbar.update()

    if iter * batch_size != test_set_size:
        big_batch_x = test_set_x[iter * batch_size:]
        for j in range(vae_num):
            batch_x = big_batch_x[:, slice_list[j]:slice_list[j + 1]]
            if gpu:
                batch_x = batch_x.cuda()
            recon, mu, log_std = vae_list[j](batch_x)
            recon = recon.cpu()
            vae_recon_list[j].append(recon.detach().numpy()[:, 0])

for i in range(cnn_num):
    cnn_recon_list[i] = np.concatenate(cnn_recon_list[i])
cnn_recon_list = np.array(cnn_recon_list).squeeze().transpose()
for i in range(vae_num):
    vae_recon_list[i] = np.concatenate(vae_recon_list[i])
vae_recon_list = np.array(vae_recon_list).squeeze().transpose()[window_size + 1:]
print(cnn_recon_list.shape, vae_recon_list.shape)
print(cnn_ground_truth.shape, vae_ground_truth.shape)
print("test set x", test_set_x.shape)
ground_truth = np.concatenate((cnn_ground_truth, vae_ground_truth), axis=1)
reconstruction_list = np.concatenate((cnn_recon_list, vae_recon_list), axis=1)

location_list = dataloader.R + dataloader.P
lut = np.zeros(len(location_list), dtype=int)
for i, index in enumerate(location_list):
    lut[index] += i
print(lut)
ground_truth = ground_truth.transpose()[lut].transpose()
reconstruction_list = reconstruction_list.transpose()[lut].transpose()

if normalize_data:
    devation, mean = dataloader.load_std_and_mean()
    print('devation,mean', devation, mean)
    ground_truth = ground_truth * devation + mean
    reconstruction_list = reconstruction_list * devation + mean
print('rebuilt ground truth\n', ground_truth[-1])

print("R&P:", dataloader.R, dataloader.P)
esp = 1e-30
score_list = np.absolute(ground_truth - reconstruction_list)
score_list_percent = np.absolute((ground_truth - reconstruction_list) / (ground_truth + esp))
print(score_list_percent)

f = open("resultpercent.csv", "w")
for i in score_list_percent:
    for j in i:
        print(j, end=',', file=f)
    print(' ', file=f)
f.close()
f = open("result.csv", "w")
for i in reconstruction_list:
    for j in i:
        print(j, end=',', file=f)
    print(' ', file=f)
f.close()
f = open("gt.csv", "w")
for i in ground_truth:
    for j in i:
        print(j, end=',', file=f)
    print(' ', file=f)
f.close()

length = 1000
y1 = ground_truth[:length, 0]
y2 = reconstruction_list[:length, 0]
x = np.arange(y1.shape[0])
plt.figure()
plt.plot(x, y1,label='ground truth')
plt.plot(x, y2,label='predicted')
plt.legend()
plt.savefig('plot.png', format='png')

y1 = ground_truth[:length, 2]
y2 = reconstruction_list[:length, 2]
x = np.arange(y1.shape[0])
plt.figure()
plt.plot(x, y1, label='ground truth')
plt.plot(x, y2, label='predicted')
plt.legend()
plt.savefig('plot2.png', format='png')

# vae_abnormal_score_list = numpy.absolute(vae_recon_list - vae_ground_truth) / vae_ground_truth
# cnn_abnormal_score_list = numpy.absolute(cnn_recon_list - cnn_ground_truth) / cnn_ground_truth
# sample_abnormal_score_list = numpy.concatenate((cnn_abnormal_score_list, vae_abnormal_score_list), axis=1)
# sample_abnormal_score_list = numpy.max(sample_abnormal_score_list, axis=1)
# print(sample_abnormal_score_list)
exit()

# train vae
'''
train_set = torch.Tensor(P_trainset_x[0])
test_set = torch.Tensor(P_testset_x[0])
input_size = train_set.shape[1]
latent_size = latent
train_set_size = train_set.shape[0]
print(input_size)

model = VAE(input_size, latent_size)
if gpu:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()

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
            recon_loss = mse_loss(recon, batch_x)
            pbar.set_postfix_str(
                "epochs: %d/%d train loss: %.6f test loss: -.------" % (epochs + 1, epoch, recon_loss.item()))
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

# train ae
if multivariate:
    R_train_set_x = torch.Tensor(R_trainset_x)
    R_test_set_x = torch.Tensor(R_testset_x)
    R_input_size = R_train_set_x.shape[1]
    R_latent_size = latent
    R_trainset_size = R_train_set_x.shape[0]

    print("???", R_trainset_x.shape)

    model_R = AE(R_input_size, R_latent_size)
    if gpu:
        model_R.cuda()
    optimizer = optim.Adam(model_R.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    for epochs in range(epoch):
        if epochs % 10 == 0:
            permutation = np.random.permutation(R_trainset_size)
            R_train_set_x = R_train_set_x[permutation]

        iter = R_trainset_size // batch_size
        with tqdm(total=iter, ascii=True) as pbar:
            pbar.set_postfix_str("epochs: --- train loss: -.------ test loss: -.------")
            for i in range(iter):
                batch_x = R_train_set_x[i * batch_size:(i + 1) * batch_size]
                if gpu:
                    device = "cuda:" + str(gpu)
                    batch_x = batch_x.cuda()
                recon = model_R(batch_x)
                optimizer.zero_grad()
                loss = model_R.loss_function(recon, batch_x)
                pbar.set_postfix_str(
                    "epochs: %d/%d train loss: %.6f test loss: -.------" % (epochs + 1, epoch, loss.item()))
                loss.backward()
                optimizer.step()
                pbar.update()

            if iter * batch_size != R_trainset_size:
                batch_x = R_train_set_x[iter * batch_size:]
                if gpu:
                    batch_x = batch_x.cuda()
                recon = model_R(batch_x)
                optimizer.zero_grad()
                loss = model_R.loss_function(recon, batch_x)
                loss.backward()
                optimizer.step()
# train cnn
else:
    print(torch.Tensor(R_trainset_x[0]).shape)
    R_train_set_x = torch.Tensor(R_trainset_x[0]).transpose(1, -1)
    R_train_set_y = torch.Tensor(R_trainset_y[0])
    R_test_set_x = torch.Tensor(R_testset_x[0]).transpose(1, -1)
    R_test_set_y = torch.Tensor(R_testset_y[0])
    R_input_size = R_train_set_x.shape[1]
    R_latent_size = latent
    R_trainset_size = R_train_set_x.shape[0]

    print(R_train_set_x.shape, R_train_set_y.shape)

    model_R = CNN(window_size)
    if gpu:
        model_R.cuda()
    optimizer = optim.Adam(model_R.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    for epochs in range(epoch):
        if epochs % 10 == 0:
            permutation = np.random.permutation(R_trainset_size)
            R_train_set_x = R_train_set_x[permutation]
            R_train_set_y = R_train_set_y[permutation]

        iter = R_trainset_size // batch_size
        with tqdm(total=iter, ascii=True) as pbar:
            pbar.set_postfix_str("epochs: --- train loss: -.------ test loss: -.------")
            for i in range(iter):
                batch_x = R_train_set_x[i * batch_size:(i + 1) * batch_size]
                batch_y = R_train_set_y[i * batch_size:(i + 1) * batch_size]
                if gpu:
                    device = "cuda:" + str(gpu)
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                # print("batchxshape",batch_x.shape)
                recon = model_R(batch_x)
                optimizer.zero_grad()
                loss = model_R.loss_function(recon, batch_y)
                pbar.set_postfix_str(
                    "epochs: %d/%d train loss: %.6f test loss: -.------" % (epochs + 1, epoch, loss.item()))
                loss.backward()
                optimizer.step()
                pbar.update()

            if iter * batch_size != R_trainset_size:
                batch_x = R_train_set_x[iter * batch_size:]
                batch_y = R_train_set_y[iter * batch_size:]
                if gpu:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                recon = model_R(batch_x)
                optimizer.zero_grad()
                loss = model_R.loss_function(recon, batch_y)
                loss.backward()
                optimizer.step()'''

# train vae


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
