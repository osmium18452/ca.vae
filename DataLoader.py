import numpy as np
import pickle
from causallearn.search.ScoreBased.GES import ges

from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
import pickle
import os


class DataLoader:
    def __init__(self, trainset_filename, testset_filename, testset_gt_filename, n_variate=None, n_samples=None,
                 data_normalization=False):
        f = open(trainset_filename, "rb")
        self.train_data = np.array(pickle.load(f))
        print("...", self.train_data.shape)
        f.close()
        self.train_data = self.train_data[:n_samples]

        f = open(testset_filename, "rb")
        self.test_data = np.array(pickle.load(f))
        print('... test data', self.test_data.shape)
        f.close()
        self.test_data = self.test_data[:n_samples]

        f = open(testset_gt_filename, "rb")
        self.test_label = np.array(pickle.load(f))
        f.close()
        self.test_label = self.test_label[:n_samples, :n_variate]

        self.zero_variate = np.where(np.sum(self.train_data, axis=0) == 0)[0]
        self.non_zero_variate = np.where(np.sum(self.train_data, axis=0) != 0)[0]

        self.non_zero_data_train = self.train_data.transpose()[self.non_zero_variate[:n_variate]].transpose()
        self.non_zero_data_test = self.test_data.transpose()[self.non_zero_variate[:n_variate]].transpose()
        # normalization
        self.train_devation = np.std(self.non_zero_data_train, axis=0)
        self.train_mean = np.mean(self.non_zero_data_train, axis=0)
        # self.test_devation = np.std(self.non_zero_data_test, axis=0)
        # self.test_mean = np.mean(self.non_zero_data_test, axis=0)
        # print('data:',self.non_zero_data_train[0], 'mean:',self.train_mean, 'devation:',self.train_devation)
        # print("ddd",(self.non_zero_data_train[0,0]-self.train_mean[0])/self.train_devation[0])
        # print(self.non_zero_data_test[0], self.test_mean, self.test_devation)
        # print(self.test_devation.shape, self.test_mean.shape, "test_devation,test_mean")
        print(self.non_zero_data_train[0])
        print(self.non_zero_data_test[0])
        # np.savetxt('original_test_set.csv',self.non_zero_data_test,delimiter=',')
        if data_normalization:
            self.non_zero_data_train = (self.non_zero_data_train - self.train_mean) / self.train_devation
            self.non_zero_data_test = (self.non_zero_data_test - self.train_mean) / self.train_devation
        print(self.non_zero_data_train[0])
        print(self.non_zero_data_test[0])
        # np.savetxt('normalized_test_set.csv',self.non_zero_data_test,delimiter=',')
        # np.savetxt('recon_test_set.csv',self.non_zero_data_test*self.train_devation+self.train_mean)

        print('non zero data test', self.non_zero_data_test.shape)
        print('non zero data train', self.non_zero_data_train.shape)

    def load_std_and_mean(self):
        return self.train_devation, self.train_mean

    def load_causal_data(self):
        return self.non_zero_data_train

    def load_zero_variate(self):
        return self.zero_variate

    def load_non_zero_variate(self):
        return self.non_zero_variate

    def load_num_non_zero_variate(self):
        return len(self.non_zero_variate)

    def get_parents(self, graph):
        nodes = graph.shape[0]
        parents_list = []
        for i in range(nodes):
            parents_list.append([])
        for i in range(nodes):
            for j in range(nodes):
                if graph[i][j] == -1:
                    parents_list[j].append(i)
        return parents_list

    def prepare_ad_data(self, graph, univariate=True, temporal=False, window_size=20):
        parent_list = self.get_parents(graph)
        self.parent_list = parent_list
        self.R = []
        self.P = []
        print(graph.shape,'graph shape')
        for (i, node) in enumerate(parent_list):
            if len(node) == 0:
                self.R.append(i)
            else:
                self.P.append(i)
        # data for R
        if univariate:
            self.R_trainset_x = [[] for i in range(len(self.R))]
            self.R_trainset_y = [[] for i in range(len(self.R))]
            self.R_testset_x = [[] for i in range(len(self.R))]
            self.R_testset_y = [[] for i in range(len(self.R))]

            X_train = self.non_zero_data_train.transpose()
            X_test = self.non_zero_data_test.transpose()
            for (i, index) in enumerate(self.R):
                for j in range(X_train.shape[-1] - window_size - 1):
                    self.R_trainset_x[i].append(X_train[index][j:j + window_size])
                    self.R_trainset_y[i].append([X_train[index][j + window_size]])
                for j in range(X_test.shape[-1] - window_size - 1):
                    self.R_testset_x[i].append(X_test[index][j:j + window_size])
                    self.R_testset_y[i].append([X_test[index][j + window_size]])
            print(X_train.shape[-1] - window_size - 1, X_train.shape[-1], len(self.R_testset_x[1]),
                  len(self.R_trainset_x[1]), "hahahaha")

            self.R_trainset_x = np.expand_dims(np.array(self.R_trainset_x),
                                               -1)  # [variate_index, sample_number, window_size,channel=1]
            self.R_trainset_y = np.array(self.R_trainset_y)  # [variate_index, ground_truth,1]
            self.R_testset_x = np.expand_dims(np.array(self.R_testset_x), -1)
            self.R_testset_y = np.array(self.R_testset_y)
            # print(self.R_trainset_x.shape, self.R_trainset_y.shape)
            # print(self.R_testset_x.shape, self.R_testset_y.shape)
        else:
            if temporal:
                pass
            else:
                self.R_trainset_x = self.non_zero_data_train.transpose()[np.array(self.R)].transpose()
                self.R_trainset_y = []
                self.R_testset_x = self.non_zero_data_test.transpose()[np.array(self.R)].transpose()
                self.R_testset_y = []
                print(self.R_trainset_x.shape)

        # data for P
        self.P_trainset_x = []  # [variate_num,sample_number,parent_number] the first is the child.
        self.P_trainset_y = []  # [variate_num,sample_number]
        self.P_testset_x = []
        self.P_testset_y = []
        print("non zero data test 2", self.non_zero_data_test.shape)

        for i in self.P:
            self.P_trainset_x.append(self.non_zero_data_train.transpose()[np.array([i] + parent_list[i])].transpose())
            # self.P_trainset_y.append(self.non_zero_data_train.transpose()[i].transpose())
            self.P_testset_x.append(self.non_zero_data_test.transpose()[np.array([i] + parent_list[i])].transpose())
            # self.P_testset_y.append(self.non_zero_data_test.transpose()[i].transpose())
            # print(parent_list[i])
            # print(self.non_zero_data_train.transpose()[np.array(parent_list[i])].transpose().shape)
        # print(np.shape(self.P_trainset_y))

    def load_train_data(self, univariate=True):
        if univariate:
            print(self.P_trainset_x[0].dtype, "load")
            return self.R_trainset_x, self.R_trainset_y, self.P_trainset_x
        else:
            return self.R_trainset_x, self.P_trainset_x

    def load_test_data(self, univariate=True):
        if univariate:
            return self.R_testset_x, self.R_testset_y, self.P_testset_x
        else:
            return self.R_testset_x, self.P_testset_x

    def load_P_input_len_list(self):
        p = []
        for i in self.parent_list:
            if len(i) > 0:
                p.append(len(i) + 1)
        return p

    def load_infernocus_train_data_P(self):
        tmp_list = [i for i in self.P_trainset_x]
        # print(self.P_trainset_x[0].dtype,"dl ")
        P_trainset_tmp = np.concatenate((tuple(tmp_list)), axis=1)
        return P_trainset_tmp

    def load_infernocus_test_data_P(self):
        tmp_list = [i for i in self.P_testset_x]
        # print('tmp list',self.P_testset_x.shape)
        # print(self.P_testset_x[0].dtype,"dl ")
        P_testset_tmp = np.concatenate((tuple(tmp_list)), axis=1)
        return P_testset_tmp

    def load_cnn_train_data(self):
        return self.R_trainset_x, self.R_trainset_y

    def load_cnn_test_data(self):
        return self.R_testset_x, self.R_testset_y


if __name__ == '__main__':
    trainset_filename = "ServerMachineDataset/train/pkl/machine-2-1.pkl"
    testset_filename = "ServerMachineDataset/test/pkl/machine-2-1.pkl"
    testset_gt_filename = "ServerMachineDataset/test_label/pkl/machine-2-1.pkl"
    dataloader = DataLoader(trainset_filename, testset_filename, testset_gt_filename, n_variate=None,
                            data_normalization=True)
    X = dataloader.load_causal_data()
    # Record = ges(X, maxP=5)
    with open("save/machine-2-1.camap.pkl", "rb") as f:
        #     pickle.dump(Record,f)
        Record = pickle.load(f)
    # print(Record['G'].graph)
    # print(Record['G'])
    dataloader.prepare_ad_data(Record['G'].graph, univariate=True)
    R_trainset_x, P_trainset_x = dataloader.load_train_data(univariate=False)
    R_testset_x, P_testset_x = dataloader.load_test_data(univariate=False)
    print(dataloader.R+dataloader.P)

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
