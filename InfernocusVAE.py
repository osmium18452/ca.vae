import torch
from torch import nn
import torch.nn.functional as F
from VAE import VAE
import numpy as np


class InfernocusVAE(nn.Module):
    def __init__(self, input_size_list, latent_size):
        self.input_size_list = input_size_list
        self.slice_list = [0,]
        for i in self.input_size_list:
            self.slice_list.append(self.slice_list[-1] + i)
        super(InfernocusVAE, self).__init__()
        self.vae_num = len(input_size_list)
        self.vae_list = []
        for i in input_size_list:
            self.vae_list.append(VAE(i, latent_size))

        # print(self.vae_list)
        # print(np.sum(self.input_size_list),self.input_size_list)
        # print(self.slice_list)

    def forward(self, x):
        input_list = []
        for i in range(self.vae_num):
            input_list.append(x[:, self.slice_list[i]:self.slice_list[i + 1]])
        out_list_recon = []
        out_list_mu = []
        out_list_log_std = []
        for i in range(self.vae_num):
            recon, mu, log_std = self.vae_list[i](input_list[i])
            out_list_recon.append(recon)
            out_list_mu.append(mu)
            out_list_log_std.append(log_std)
        ret_recon = torch.cat(tuple(out_list_recon), 1)
        ret_mu = torch.cat(tuple(out_list_mu), 1)
        ret_log_std = torch.cat(tuple(out_list_log_std), 1)
        # for i in out_list_log_std:
        #     print(i.shape)
        return ret_recon, ret_mu, ret_log_std

    def loss_function(self, recon, x, mu, log_std):
        loss_list = []
        for i in range(self.vae_num):
            recon_loss = F.mse_loss(recon[:, self.slice_list[i]:self.slice_list[i + 1]],
                                    x[:, self.slice_list[i]:self.slice_list[i + 1]],
                                    reduction="sum")  # use "mean" may have a bad effect on gradients
            kl_loss = -0.5 * (1 + 2 * log_std[:, self.slice_list[i]:self.slice_list[i + 1]] -
                              mu[:, self.slice_list[i]:self.slice_list[i + 1]].pow(
                                  2) - torch.exp(2 * log_std[:, self.slice_list[i]:self.slice_list[i + 1]]))
            kl_loss = torch.sum(kl_loss)
            loss_list.append(recon_loss + kl_loss)
        loss_list=torch.Tensor(loss_list)
        # print("loss list",loss_list.shape)
        return torch.mean(loss_list)
            # mse_list.append(F.mse_loss(recon[:, self.slice_list[i]:self.slice_list[i + 1]],
            #                            x[:, self.slice_list[i]:self.slice_list[i + 1]]))
            # kl_loss=

# [ 2  3  4  5  6  7  8  9  0 10 11 12 13 14 15  1 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]