import pickle
import numpy as np
import faiss

data = pickle.load(open('./output/data.pkl','rb'))
tf_discrete = pickle.load(open('./output/tf_discrete.pkl','rb'))

tf = np.array([tf_discrete[d['name']] for d in data]).astype(np.float32)
np.save('./output/tf_discrete.npy',tf)

d = 1000
tf = tf.astype(np.float32)[:1000]

ncentroids = 1000
niter = 200
verbose = True

kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose,gpu=True)
kmeans.train(tf)
centroids = kmeans.centroids

index = faiss.IndexFlatL2(d)
index.add(tf)
nNN = 1000
D, I = index.search (kmeans.centroids, nNN)

np.save(f'./output/tf_centroids.npy',centroids)
np.save(f'./output/tf_clusters.npy',I)
