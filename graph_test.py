import numpy as np

def convert_to_float(words):
    ret = []
    for i in words:
        ret.append(float(i))
    return ret

f=open("testdata.txt","r")
data=[]
for line in f.readlines():
    words=line.strip().split()
    data.append(convert_to_float(words))
data=np.array(data)
# print(data.shape)

from causallearn.search.ScoreBased.GES import ges

Record = ges(data)

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

pyd = GraphUtils.to_pydot(Record['G'])
print("**********************")
print(Record['G'].graph)
print(Record['G'])


tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()


# # or save the graph
# pyd.write_png('simple_test.png')