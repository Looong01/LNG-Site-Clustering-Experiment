import torch

class KMEANS_CUDA:
    def __init__(self, n_clusters, distance_function = "cos", iter=1, batch_size=0, thresh=1e-5, norm_center=False):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.dists = float("inf")  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.cluster_centers_ = None
        self.iter=iter
        self.batch_size=batch_size
        self.thresh=thresh
        self.norm_center=norm_center
        if distance_function == "l2":
            self.distance_function = self.l2_distance
        elif distance_function == "cos":
            self.distance_function = self.cosine_distance
    
    def fit(self, X):
        obs = torch.from_numpy(X).cuda()
        torch.cuda.synchronize()
        for i in range(self.iter):
            if self.batch_size == 0:
                self.batch_size == obs.shape[0]
            labels, centers, distance = self._kmeans_batch(obs,
                                            self.n_clusters,
                                            self.norm_center,
                                            self.distance_function,
                                            self.batch_size,
                                            self.thresh)
            self.labels_ = labels.cpu().numpy()
            if distance < self.dists:
                self.centers = centers
                self.dists = distance
                self.cluster_centers_ = centers.cpu().numpy()
        torch.cuda.synchronize()
        return self

    def _kmeans_batch(self, obs: torch.Tensor, k: int, norm_center, distance_function, batch_size, thresh):
        # k x D
        centers = obs[torch.randperm(obs.size(0))[:k]].clone()
        history_distances = [float('inf')]
        if batch_size == 0:
            batch_size = obs.shape[0]
        while True:
            # (N x D, k x D) -> N x k
            segs = torch.split(obs, batch_size)
            seg_center_dis = []
            seg_center_ids = []
            for seg in segs:
                distances = distance_function(seg, centers)
                center_dis, center_ids = distances.min(dim=1)
                seg_center_ids.append(center_ids)
                seg_center_dis.append(center_dis)
            obs_center_dis_mean = torch.cat(seg_center_dis).mean()
            obs_center_ids = torch.cat(seg_center_ids)
            history_distances.append(obs_center_dis_mean.item())
            diff = history_distances[-2] - history_distances[-1]
            if diff < thresh:
                if diff < 0:
                    continue
                break
            for i in range(k):
                obs_id_in_cluster_i = obs_center_ids == i
                if obs_id_in_cluster_i.sum() == 0:
                    continue
                obs_in_cluster = obs.index_select(0, obs_id_in_cluster_i.nonzero().squeeze())
                c = obs_in_cluster.mean(dim=0)
                if norm_center:
                    c /= c.norm()
                centers[i] = c
        return obs_center_ids, centers, history_distances[-1]

    def cosine_distance(self, obs, centers):
        obs_norm = obs / obs.norm(dim=1, keepdim=True)
        centers_norm = centers / centers.norm(dim=1, keepdim=True)
        cos = torch.matmul(obs_norm, centers_norm.transpose(1, 0))
        return 1 - cos

    def l2_distance(self, obs, centers):
        dis = ((obs.unsqueeze(dim=1) - centers.unsqueeze(dim=0)) ** 2.0).sum(dim=-1).squeeze()
        return dis