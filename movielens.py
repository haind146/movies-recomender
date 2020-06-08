import pandas as pd
import numpy as np
from CF import CF
from clustering import bisecting_kmeans2
from sklearn.cluster import KMeans


ratings_base = pd.read_csv('data-split/train.csv')
ratings_test = pd.read_csv('data-split/test.csv')


rate_train = ratings_base.values
rate_test = ratings_test.values


# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

rs = CF(rate_train, None, k=30, uuCF=1)

rs.normalize_Y()
rs.similarity()
print('clustering...')
clusters, user_mapping = bisecting_kmeans2(rs.Ybar.transpose().tocsr(), 40)
print('clustering done')
for i in clusters:
    print('cluster_id:', i, ':', clusters[i].get_shape()[0])
cf_clusters = {}
print(clusters)
for i in clusters:
    cf_clusters[i] = CF(None, clusters[i].transpose(), k=500)
    print('calculate similarity cluster', i)
    cf_clusters[i].similarity()

n_tests = rate_test.shape[0]
SE = 0 # squared error
for n in range(n_tests):
    if n % 1000 == 0:
        print(n)
    cf_cluster = cf_clusters[user_mapping[rate_test[n, 0]][0]]
    user_cluster_id = user_mapping[rate_test[n, 0]][1]
    pred = cf_cluster.pred(user_cluster_id, rate_test[n, 1], normalized=1) + rs.mu[rate_test[n, 0]]
    SE += (pred - rate_test[n, 2])**2

print(SE)
RMSE = np.sqrt(SE/n_tests)
print('User-user CF, RMSE =', RMSE)

