import os

from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
import pickle
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_train_samples", default=-1, type=int)
parser.add_argument("-s", "--save_name", default="result.pkl", type=str)
parser.add_argument("-p", "--parent", default=5, type=int)
args = parser.parse_args()

num_train_samples=args.num_train_samples
save_name=args.save_name
parent=args.parent

train_data_path="ServerMachineDataset/train/pkl"
train_data_file_name="machine-1-1.pkl"
f=open(os.path.join(train_data_path,train_data_file_name),"rb")
data=pickle.load(f)
f.close()
print(data.shape)


X=data[:num_train_samples]
print("getting graph")
record = pc(X)
print("finished")

if not os.path.exists("save"):
    os.makedirs("save")
f= open(os.path.join("save",save_name),"wb")
pickle.dump(record,f)
f.close()

record.draw_pydot_graph()
# pyd = GraphUtils.to_pydot(Record['G'])
# tmp_png = pyd.create_png(f="png")
# fp = io.BytesIO(tmp_png)
# img = mpimg.imread(fp, format='png')
# plt.axis('off')
# plt.imshow(img)
# plt.show()

#
# # or save the graph
# pyd.write_png('simple_test.png')