from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def predict(i, clustering, data):
    labels = clustering.labels_
    centers = [] 
    one_cluster = data[labels == i]  
    fig = plt.figure()
    ax = Axes3D(fig)
    print("num of points " + str(one_cluster.shape[0]))
    print("-1:discharge 1:load 0:no action")
    print(one_cluster[:,2])
    center = np.average(one_cluster,axis=0)
    print(center)
    # centers.append(center.tolist())
    load_rate = (one_cluster[:,2]>0).sum()/one_cluster.shape[0]
    discharge_rate = (one_cluster[:,2]<0).sum()/one_cluster.shape[0]
    spare_rate = (one_cluster[:,2]==0).sum()/one_cluster.shape[0]
    if(discharge_rate>=0.1 and discharge_rate>load_rate):
        print("import")
    elif(load_rate>=0.1 and load_rate>discharge_rate):
        print("export")
    else:
        print("mooring")
    ax.scatter(one_cluster[:,0],one_cluster[:,1],one_cluster[:,2])
    ax.scatter(center[0],center[1],center[2])