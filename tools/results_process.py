import matplotlib.pyplot as plt
import numpy as np
import json
import os
import fileinput

def results_process(clustering, data, algorithm, filename="lng_results_list.json"):
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    dic ={
    "import":[],
    "export":[],
    "mooring":[] 
    }
    save_dic = {
    # "code": i , "latitude":xxx, "longitude":xxx, "isLNG":xxxx, "IN":xxxx
    }
    
    cwd = os.getcwd()
    if not os.path.exists(cwd + "/json"):
        os.makedirs(cwd + "/json")

    json_path = "json/" + filename

    with open(json_path, 'w') as f:
        f.write('{\n')
    
    centers = []

    plt.figure(figsize=(9,7),dpi=100)
    plt.title(algorithm)

    for i in range(n_clusters_):
        one_cluster = data[labels == i]
        load_rate = (one_cluster[:,2]>0).sum()/one_cluster.shape[0]
        discharge_rate = (one_cluster[:,2]<0).sum()/one_cluster.shape[0]
        spare_rate = (one_cluster[:,2]==0).sum()/one_cluster.shape[0] 
        center = np.average(one_cluster,axis=0)
        centers.append(center.tolist())
        save_dic["code"] = i + 1
        save_dic["latitude"] = center[1]
        save_dic["longitude"] = center[0]
        if(discharge_rate>=0.05 and discharge_rate>load_rate):
            dic["import"].append(center)
            save_dic["isLNG"] = True
            save_dic["IN"] = True
            type1 = plt.scatter(center[0],center[1], c='r', s=5)  
        elif(load_rate>=0.05 and load_rate>discharge_rate):
            dic["export"].append(center)
            save_dic["isLNG"] = True
            save_dic["IN"] = False
            type2 = plt.scatter(center[0],center[1], c='b', s=5)
        else:
            dic["mooring"].append(center)
            save_dic["isLNG"] = False
            save_dic["IN"] = None
            type3 = plt.scatter(center[0],center[1], c='g', s=5)
        with open(json_path, 'a') as f:
            f.write('\t')
        save_json(save_dic, json_path)
        with open(json_path, 'a') as f:
            f.write('\n')
    plt.legend((type1, type2, type3), (u'Import', u'Export', u'Mooring'))
    plt.savefig("output/Cluster_Centers_" + algorithm + ".png")
    with open(json_path, 'a') as f:
        f.write('}')
    for line in fileinput.input(json_path, inplace=1):
        line=line.replace("null", "None")
        print(line,end="")
    return dic

def save_json(data, json_path):
    with open(json_path, 'a') as f:
        json.dump(data, f)