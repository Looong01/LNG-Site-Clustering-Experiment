#导入工具包
from preprocess import preprocess, distance, delete_file
from data_loader import data_loader
from results_process import results_process
from predict import predict
#使用sklearn
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering

#数据预处理
preprocess("lng2.csv", "Ing2_stop_spots.csv")

#数据加载
loader = data_loader("Ing2_stop_spots.csv")
data, coord = loader.load()
loader.show_scatter()

#使用K-means算法聚合数据
clustering_kmeans = KMeans(n_clusters=300, init='random').fit(coord)
dic = results_process(clustering_kmeans, data, "K-means", filename="lng_results_list(K-means).json")
print("K-means")
print("总数: ", len(clustering_kmeans.labels_))
print("聚类中心点数: ", len(clustering_kmeans.cluster_centers_))
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")

#使用K-means++算法聚合数据
clustering_kmeanspp = KMeans(n_clusters=300, init='k-means++').fit(coord)
dic = results_process(clustering_kmeanspp, data, "K-means++", filename="lng_results_list(K-means++).json")
print("K-means++")
print("总数: ", len(clustering_kmeanspp.labels_))
print("聚类中心点数: ", len(clustering_kmeanspp.cluster_centers_))
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")

#使用DBSCAN算法聚合数据
clustering_dbscan = DBSCAN(eps=1500, min_samples=6, n_jobs=-1,metric=distance).fit(data)
dic = results_process(clustering_dbscan, data, "DBSCAN", filename="lng_results_list(DBSCAN).json")
print("DBSCAN")
print("总数: ", len(clustering_dbscan.labels_))
print("聚类中心点数: ", len(set(clustering_dbscan.labels_)) - (1 if -1 in clustering_dbscan.labels_ else 0))
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")

#使用OPTICS算法聚合数据
clustering_optics = OPTICS(min_samples=20).fit(data)
dic = results_process(clustering_optics, data, "OPTICS", filename="lng_results_list(OPTICS).json")
print("OPTICS")
print("总数: ", len(clustering_optics.labels_))
print("聚类中心点数: ", len(set(clustering_optics.labels_)) - (1 if -1 in clustering_optics.labels_ else 0))
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")

#使用AGNES算法聚合数据
clustering_agnes = AgglomerativeClustering(n_clusters=300).fit(coord)
dic = results_process(clustering_agnes, data, "AGNES", filename="lng_results_list(AGNES).json")
print("AGNES")
print("总数: ", len(clustering_agnes.labels_))
print("聚类中心点数: ", clustering_agnes.n_clusters_)
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")

#删除预处理文件
delete_file("lng2_stop_spots.csv")