#导入工具包
from preprocess import preprocess, distance, delete_file
from data_loader import data_loader
from results_process import results_process
from predict import predict
from tools.kmeans_CUDA import KMEANS_CUDA
#使用sklearn
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
#加载时间工具
import time

#数据预处理
start = time.time()
preprocess("lng2.csv", "lng2_stop_spots.csv")
end = time.time()
print("耗时：", end - start, "秒")

#数据加载
start = time.time()
loader = data_loader("lng2_stop_spots.csv")
data, coord = loader.load()
loader.show_scatter()
end = time.time()
print("耗时：", end - start, "秒")

#使用K-means算法聚合数据
start = time.time()
clustering_kmeans = KMeans(n_clusters=400, init='random').fit(coord)
end = time.time()
dic = results_process(clustering_kmeans, data, "K-means", filename="lng_results_list(K-means).json")
print("K-means")
print("总数: ", len(clustering_kmeans.labels_))
print("聚类中心点数: ", len(clustering_kmeans.cluster_centers_))
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")
print("耗时：", end - start, "秒")

#使用K-means++算法聚合数据
start = time.time()
clustering_kmeanspp = KMeans(n_clusters=600, init='k-means++').fit(coord)
end = time.time()
dic = results_process(clustering_kmeanspp, data, "K-means++", filename="lng_results_list(K-means++).json")
print("K-means++")
print("总数: ", len(clustering_kmeanspp.labels_))
print("聚类中心点数: ", len(clustering_kmeanspp.cluster_centers_))
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")
print("耗时：", end - start, "秒")

#使用K-means算法在GPU上聚合数据
start = time.time()
clustering_kmeans_cuda = KMEANS_CUDA(n_clusters=400).fit(coord)
end = time.time()
dic = results_process(clustering_kmeans_cuda, data, "K-means_CUDA", filename="lng_results_list(K-means_CUDA).json")
print("K-means_CUDA")
print("总数: ", len(clustering_kmeans_cuda.labels_))
print("聚类中心点数: ", len(clustering_kmeans_cuda.cluster_centers_))
print()
print("其中，")
print("\tLNG入口点有: ", len(dic["import"]), "个")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")
print("耗时：", end - start, "秒")

#使用DBSCAN算法聚合数据
start = time.time()
clustering_dbscan = DBSCAN(eps=1000, min_samples=3, n_jobs=-1,metric=distance).fit(data)
end = time.time()
dic = results_process(clustering_dbscan, data, "DBSCAN", filename="lng_results_list(DBSCAN).json")
print("DBSCAN")
print("总数: ", len(clustering_dbscan.labels_))
print("聚类中心点数: ", len(set(clustering_dbscan.labels_)) - (1 if -1 in clustering_dbscan.labels_ else 0))
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")
print("耗时：", end - start, "秒")

#使用OPTICS算法聚合数据
start = time.time()
clustering_optics = OPTICS(min_samples=10).fit(data)
end = time.time()
dic = results_process(clustering_optics, data, "OPTICS", filename="lng_results_list(OPTICS).json")
print("OPTICS")
print("总数: ", len(clustering_optics.labels_))
print("聚类中心点数: ", len(set(clustering_optics.labels_)) - (1 if -1 in clustering_optics.labels_ else 0))
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")
print("耗时：", end - start, "秒")

#使用AGNES算法聚合数据
start = time.time()
clustering_agnes = AgglomerativeClustering(n_clusters=650).fit(coord)
end = time.time()
dic = results_process(clustering_agnes, data, "AGNES", filename="lng_results_list(AGNES).json")
print("AGNES")
print("总数: ", len(clustering_agnes.labels_))
print("聚类中心点数: ", clustering_agnes.n_clusters_)
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")
print("耗时：", end - start, "秒")

#删除预处理文件
delete_file("lng2_stop_spots.csv")