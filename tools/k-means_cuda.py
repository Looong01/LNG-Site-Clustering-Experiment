import torch
import time
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot
class KMEANS:
    def __init__(self, n_clusters=20, max_iter=None, verbose=True,device = torch.device("cpu")):

        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = x[init_row]
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def representative_sample(self):
        # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
        self.representative_samples = torch.argmin(self.dists, (0))
    
    def show(self):
        centers = self.centers.cpu().numpy()
        # labels = self.labels.cpu().numpy()
        print(centers)
        # for i in range(len(labels)):
        #     pyplot.scatter(coord[i][0], coord[i][1], c=('r' if labels[i] == 0 else 'b'))
        pyplot.scatter(centers[:,0],centers[:,1],marker='*', s=100)
        pyplot.savefig(str(self.n_clusters) + '.png')
def RunKMEANS(matrix,device,clusters):
    k = KMEANS(n_clusters=clusters, max_iter=10,verbose=False,device=device)
    k.fit(matrix)
    k.show()

def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device
cwd = os.getcwd()
print(cwd)
path = cwd + "/lng3.csv"
new_path = cwd + "/lng_low_velocity.csv"
csv = pd.read_csv(path)

data = {'id':[],
           'time':[],
           'status':[],
           'velocity':[],
           'long':[],
           'lati':[],
           'draft':[]}
attri_list = ['id','time','status','velocity','long','lati','draft']

coord = np.array(csv[['long','lati']])
for i in range(len(csv)):
    coord[i] = np.array([csv['long'][i],csv['lati'][i]])
print(coord)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure()

    device = choose_device(True)
    
    # coord = np.array([ [1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11] ])

    pyplot.scatter(coord[:,0], coord[:,1], s=0.3)
    
    matrix = torch.from_numpy(coord).to(device)
    RunKMEANS(matrix, device, 400)