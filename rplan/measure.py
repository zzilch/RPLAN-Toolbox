import numpy as np

def compute_tf(boundary):
    '''
    input: boundary points array (x,y,dir,isNew)
    return: tf.x, tf.y
    '''
    if boundary.shape[1]>2:
        boundary=boundary[:,:2]
    boundary = np.concatenate((boundary,boundary[:1]))
    num_point = len(boundary)-1
    line_vector = boundary[1:]-boundary[:-1]
    line_length = np.linalg.norm(line_vector,axis=1)
    
    perimeter = line_length.sum()
    line_vector = line_vector/perimeter
    line_length = line_length/perimeter

    angles = np.zeros(num_point)
    for i in range(num_point):
        z = np.cross(line_vector[i],line_vector[(i+1)%num_point])
        sign = np.sign(z)
        angles[i] = np.arccos(np.dot(line_vector[i],line_vector[(i+1)%num_point]))*sign

    x = np.zeros(num_point+1)
    y = np.zeros(num_point+1)
    s = 0
    for i in range(1,num_point+1):
        x[i] = line_length[i-1]+x[i-1]
        y[i-1] = angles[i-1]+s
        s = y[i-1]
    y[-1] = s
    return x,y

def sample_tf(x,y,ndim=1000):
    '''
    input: tf.x,tf.y, ndim
    return: n-dim tf values
    '''
    t = np.linspace(0,1,ndim)
    return np.piecewise(t,[t>=xx for xx in x],y)

class TFRetriever():
    def __init__(self,tf,tf_centroids,tf_clusters):
        '''
        tf: np.arraay, tf of training data
        tf_centroids: np.arraay, tf cluster tf_centroids of training data
        tf_clusters: np.arraay, data index for each cluster of training data
        '''
        self.tf = tf
        self.tf_centroids = tf_centroids
        self.tf_clusters = tf_clusters
    
    def retrieve_bf(self,boundary,k=20):
        # compute tf for the data boundary
        x,y = compute_tf(boundary)
        y_sampled = sample_tf(x,y,1000)
        dist = np.linalg.norm(y_sampled-self.tf,axis=1)
        if k>np.log2(len(self.tf)):
            index = np.argsort(dist)[:k]
        else:
            index = np.argpartition(dist,k)[:k]
            index = index[np.argsort(dist[index])]
        return index

    def retrieve_cluster(self,boundary,k=20,beam_search=False):
        '''
        boundary: test boundary
        k: retrieval num
        return: index for training data 
        '''
        # compute tf for the data boundary
        x,y = compute_tf(boundary)
        y_sampled = sample_tf(x,y,1000)
        # compute distance to cluster centers
        dist = np.linalg.norm(y_sampled-self.tf_centroids,axis=1)

        if beam_search:
            # more candicates
            c = int(np.max(np.clip(np.log2(k),1,5)))
            cluster_idx = np.argsort(dist)[:c]
            cluster = np.unique(self.tf_clusters[cluster_idx].reshape(-1))
        else:
            # only candicates
            cluster_idx = np.argmin(dist)
            cluster = self.tf_clusters[cluster_idx]

        # compute distance to cluster samples
        dist = np.linalg.norm(y_sampled-self.tf[cluster],axis=1)
        index = cluster[np.argsort(dist)[:k]]
        return index
