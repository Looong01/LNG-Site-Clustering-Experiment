import os
import gc
import math
import pandas as pd

def distance(a,b):  
    pi = 3.14
    dis = 6371000 * math.acos(math.cos(a[1]*pi/180) * math.cos(b[1]*pi/180) * math.cos((a[0]-b[0])*pi/180) + math.sin(a[1]*pi/180) * math.sin(b[1]*pi/180)-1e-12)
    return dis

def preprocess(lng2, lng_stop_spots):
    cwd = os.getcwd()
    lng2_path = cwd + '/data/' +  lng2
    temp_path = cwd + '/data/temp.csv'
    lng_stop_spots_path = cwd + '/data/' + lng_stop_spots

    #转换假csv文件为真csv文件
    content = open(lng2_path)
    with open(temp_path, "w") as f:
        for line in content:
            f.write(line.replace(" ", ","))
    with open(temp_path, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('id,time,status,velocity,long,lati,draft\n' + content)
    
    #筛选低速数据
    csv = pd.read_csv(temp_path)
    data = {
        'id':[],
            'time':[],
            'status':[],
            'velocity':[],
            'long':[],
            'lati':[],
            'draft':[]
    }
    attri_list = ['id','time','status','velocity','long','lati','draft']
    for i in range(len(csv)):
        if csv['status'][i] == 1 or csv['status'][i] == 5 or csv['status'][i] == 15:
            for attri in attri_list:
                data[attri].append(csv[attri][i])
    df = pd.DataFrame(data)
    df.to_csv(temp_path,index=False,encoding="utf-8")
    del data
    del csv
    del df
    del attri_list
    gc.collect()

    #聚合同位数据
    csv = pd.read_csv(temp_path)
    data = {
            'long':[],
            'lati':[],
            'behavior':[]
    }
    attri_list = ['long','lati','draft']
    id = "0"
    avg_pt = [0,0]
    tem_pt = [csv[attri][0] for attri in attri_list]
    count = 0
    draft_valid = []
    for i in range(len(csv)):   
        pt = [csv[attri][i] for attri in attri_list]
    
        if distance(pt,tem_pt) < 5000:
            avg_pt[0] += pt[0]
            avg_pt[1] += pt[1]
            if pt[2] != 0:
                draft_valid.append(pt[2])
            tem_pt = pt
            count += 1
        else :
            if count != 0:
                data['long'].append(avg_pt[0] / count)
                data['lati'].append(avg_pt[1] / count)
                if len(draft_valid)>0:
                    dif = draft_valid[-1]-draft_valid[0]
                    if(dif>8):
                        data['behavior'].append(1)
                    elif(dif<-8):
                        data['behavior'].append(-1)
                    else:
                        data['behavior'].append(0)
                else:
                    data['behavior'].append(0)
            count = 0
            avg_pt = [0,0]
            tem_pt = pt
            draft_valid = []
    df = pd.DataFrame(data)
    df.to_csv(lng_stop_spots_path,index=False,encoding="utf-8")
    del data
    del csv
    del df
    del attri_list
    gc.collect()
    os.remove(temp_path)