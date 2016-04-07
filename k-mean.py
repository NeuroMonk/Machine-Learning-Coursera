import numpy as np

class KMean():
    @classmethod
    def random_init(cls, X, K):
        r = np.random.randint(len(X), size=K)
        rand_array = np.zeros((0, 3), float)
        for k in range(K):
            rand_array = np.vstack([rand_array, X[r[k]]]) 
        return rand_array

    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.nu = self.random_init(X, K)
        self.idx = np.zeros((self.X.shape[0], self.nu.shape[0]))

    def find_closest_centroids(self):
        distance = np.zeros((0, self.X.shape[0]), float)
        for x in range(self.nu.shape[0]):
                vec = self.X - self.nu[x]
                dis = np.sqrt(np.sum(np.square(vec), axis=1))
                distance = np.vstack([distance, dis])
        distance = np.transpose(distance)
        self.idx = np.argmin(distance, 1)

    def compute_mean(self):
        for k in range(self.nu.shape[0]): 
            temp = (self.idx == k) * 1
            m = sum(temp)
            temp = temp.reshape((len(temp), 1))
            temp = np.multiply(self.X, temp)
            self.nu[k] = (1./m) * sum(temp)

