import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

class Agglomeration:
    __slots__ = 'data', 'min_distance', 'clusters', 'cluster_data', 'clusters_num'

    def __init__(self, filename):
        self.read_data(filename)
        self.clusters_num = 850
        self.clusters = [[i] for i in range(self.clusters_num)]
        self.cluster_data = None

    def read_data(self, filename):
        self.data = pd.read_csv(filename, delimiter=',')
        self.data = self.data.iloc[:,1:]


    def distance(self,v1, v2):
        """

        returns euclidean distance

        :param v1:
        :param v2:
        :return:
        """
        return np.linalg.norm(v1 - v2)

    def get_matrix(self, tot_clusters):
        """
        returns Matrix with M[i,j] being the distance between ith and jth vector

        :param tot_clusters:
        :return:
        """
        self.min_distance = float('inf')
        M = [[0 for _ in range(tot_clusters)] for __ in range(tot_clusters)]

        for i in range(len(M)):
            for j in range(i, len(M)):
                if i == j:
                    continue
                M[i][j] = self.distance(self.cluster_data[i], self.cluster_data[j])
                if M[i][j] < self.min_distance:
                    self.min_distance = M[i][j]
        return M

    def updated_cluster(self, vec1, vec2):
        """
        Update cluster with average of the two vectors

        :param vec1:
        :param vec2:
        :return:
        """
        new_vec = []
        for val1, val2 in zip(vec1, vec2):
            t = ((val1 + val2) / 2)
            new_vec.append(t)

        return pd.Series(new_vec).values

    def get_correlation(self):
        """
        Return Correlation Matrix

        :return:
        """
        data = self.data.copy()
        correlation_matrix = np.corrcoef(data[1:])
        return correlation_matrix

    def run_aglomretive_clustering(self):
        """

        Driver Function which runs till total clusters are greater than 6
        1. It gets the matrix
        2. gets 2 vectors with smallest distance
        3. Merges the members of bigger cluster to smaller
        4. Updates the two vectors by taking the average

        """
        self.cluster_data = [self.data.iloc[i,:].values for i in range(self.clusters_num)]
        tot_clusters = self.clusters_num

        while(tot_clusters > 6):
            matrix = self.get_matrix(tot_clusters)
            indexes = np.where(matrix == self.min_distance)

            index_vec1_ind = indexes[0][0]
            index_vec2_ind = indexes[1][0]

            new_vec = self.updated_cluster(self.cluster_data[index_vec1_ind], self.cluster_data[index_vec2_ind])

            for x in self.clusters[index_vec2_ind]:
                self.clusters[index_vec1_ind].append(x)

            self.cluster_data[index_vec1_ind] = new_vec
            self.cluster_data.pop(index_vec2_ind)
            self.clusters.pop(index_vec2_ind)

            tot_clusters = len(self.clusters)

        for cluster_center, cluster_members in zip(self.cluster_data, self.clusters):
            print(f"cluster_center is {cluster_center}")
            print(f"cluster members count is {len(cluster_members)}")
            print((f"cluster_members are {cluster_members}"))


    def plot_dendogram(self):
        """
        Plot dendogram using Matplotlib

        :return:
        """
        data = self.data.iloc[:self.clusters_num,:]
        Z = linkage(data, 'single')
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)
        plt.show()

    def kmeans_clustering(self):
        """

        Perform Kmeans clustering using Scikit learn

        """
        X = self.data
        kmeans = KMeans(n_clusters=6)
        kmeans.fit(X)
        c=kmeans.labels_
        cluster_5 = np.where(c == 5)
        cluster_4 = np.where(c == 4)
        cluster_3 = np.where(c == 3)
        cluster_2 = np.where(c == 2)
        cluster_1 = np.where(c == 1)
        cluster_0 = np.where(c == 0)

        print(cluster_5)
        print(len(cluster_5[0]))
        print(cluster_4)
        print(len(cluster_4[0]))
        print(cluster_2)
        print(len(cluster_2[0]))
        print(cluster_1)
        print(len(cluster_1[0]))
        print(cluster_0)
        print(len(cluster_0[0]))
        print(cluster_3)
        print(len(cluster_3[0]))
