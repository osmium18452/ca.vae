import os
import numpy as np
import pickle


def convert_to_float(words):
    ret = []
    for i in words:
        ret.append(float(i))
    return ret


train_path = "ServerMachineDataset/train"
test_path = "ServerMachineDataset/test"
label_path = "ServerMachineDataset/test_label"

# file_name = "machine-1-1.txt"

for path_name in (test_path,label_path):
    for root, path, file_list in os.walk(path_name):
        for file_name in file_list:
            f = open(os.path.join(path_name, file_name), "r")

            data = []
            for line in f.readlines():
                words = line.strip().split(",")
                data.append(convert_to_float(words))
            data = np.array(data)

            f.close()
            print(data)

            f=open(os.path.join(path_name,file_name[:-4]+".pkl"),"wb")
            pickle.dump(data,f)
            f.close()
