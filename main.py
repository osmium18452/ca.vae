import os

from causallearn.search.ScoreBased.GES import ges

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
import pickle
import os
import numpy as np

train_data_path="ServerMachineDataset/train"
train_data_file_name="machine-1-1.pkl"
f=open(os.path.join(train_data_path,train_data_file_name),"rb")
data=pickle.load(f)
f.close()
print(data.shape)

X=data
print("getting graph")
Record = ges(X, maxP=10)
print("finished")
pyd = GraphUtils.to_pydot(Record['G'])
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()

#
# # or save the graph
# pyd.write_png('simple_test.png')