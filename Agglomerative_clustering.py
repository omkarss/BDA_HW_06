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
        return np.linalg.norm(v1 - v2)

    def get_matrix(self, tot_clusters):
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
        new_vec = []
        for val1, val2 in zip(vec1, vec2):
            t = ((val1 + val2) / 2)
            new_vec.append(t)

        return pd.Series(new_vec).values


    def run_aglomretive_clustering(self):
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

            print(self.clusters)
            tot_clusters = len(self.clusters)

        print(self.clusters)
        print(self.cluster_data)

    def plot_dendogram(self):
        data = self.data.iloc[:self.clusters_num,:]
        Z = linkage(data, 'single')
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)
        plt.show()

        pass

    def kmeans_clustering(self):

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

    def get_all_cluster_counts(self):
        clusters = [[0, 782, 1, 109, 295, 464, 5, 11, 664, 30, 167, 415, 113, 121, 127, 585, 389, 420, 225, 479, 149, 638, 66, 748,276, 362, 484, 564, 717, 123, 408, 733, 675, 101, 370, 551, 738, 848, 194, 116, 119, 462, 553, 346, 687, 277, \
          444, 459, 775, 103, 359, 467, 228, 527, 40, 201, 230, 510, 142, 56, 12, 497, 76, 762, 758, 792, 371, 582, 473,
          392, 756, 517, 52, 163, 252, 476, 284, 612, 363, 725, 799, 580, 144, 718, 593, 386, 535, 37, 89, 124, 849,
          423, 703, 732, 344, 818, 666, 411, 283, 478, 374, 489, 150, 388, 151, 293, 198, 790, 83, 689, 813, 338, 81,
          646, 730, 407, 257, 773, 321, 700, 626, 387, 302, 597, 272, 279, 604, 697, 786, 731, 521, 746, 258, 380, 780,
          145, 356, 747, 774, 845, 477],

         [2, 333, 44, 682, 803, 156, 82, 465, 48, 173, 315, 254, 95, 483, 80, 529, 188, 610, 678, 595,
          61, 588, 801, 86, 355, 518, 817, 106, 587, 622, 755, 168, 663, 112, 396, 468, 618, 137, 206, 463,
          486, 796, 834, 528, 543, 171, 414, 560, 110, 327, 557, 568, 63, 310, 674, 125, 514, 634, 98, 354, 300,
          719, 523, 220, 135, 304, 442, 635, 639, 550, 547, 159, 652, 136, 628, 736, 141, 298, 734, 804, 668, 251,
          720, 439, 187, 519, 795, 577, 808, 835, 699, 559, 248, 404, 317, 93, 232, 241, 637, 713, 688, 766, 36, 162,
          208, 87, 421, 301, 485, 617, 347, 160, 27, 772, 200],

         [3, 269, 449, 14, 64, 288, 353, 139, 600, 754, 520, 212, 491, 531, 42, 275, 828, 708,
          836, 800, 660, 250, 416, 742, 345, 723, 102, 169, 623, 264, 684, 841, 31, 297, 589, 797,
          34, 649, 340, 308, 195, 79, 268, 351, 554, 204, 760, 592, 764, 43, 319, 456, 176, 475, 322,
          369, 115, 698, 556, 72, 153, 207, 19, 410, 741, 237, 60, 267, 570, 400, 172, 667, 575, 238, 609, 175,
          789, 807, 352, 140, 289, 793, 437, 211, 412, 222, 546, 548, 132, 146, 331, 616, 111, 273, 591, 681, 218,
          686, 791, 263, 90, 576, 728, 653, 45, 662, 596, 221, 526, 802, 590, 311, 825, 10, 768, 199, 391, 499, 253, 287, 709],

         [4, 155, 839, 8, 67, 794, 217, 692, 581, 329, 767, 501, 39, 438, 669, 186, 446, 644, 712, 364, 648, 431, 138, 453, 246,
          309, 457, 563, 117, 770, 164, 189, 358, 516, 20, 744, 749, 524, 647, 21, 97, 323, 38, 492, 558, 46, 469, 537, 108, 152,
          450, 683, 451, 837, 84, 130, 193, 673, 68, 192, 606, 59, 598, 696, 350, 435, 482, 694, 471, 640, 305, 235, 613, 398, 727,
          460, 729, 233, 104, 299, 540, 335, 434, 88, 318, 348, 99, 229, 605, 672, 154, 382, 620, 599, 58, 53, 306, 373, 810, 73, 191,
          219, 92, 454, 214, 216, 429, 425, 750, 583, 320, 261, 538, 819, 820, 57, 185, 643, 533, 197, 203, 493, 821, 816, 349, 7, 210,
          574, 15, 641, 658, 292, 181, 657, 846, 651, 85, 809, 566, 270, 642, 665, 721, 502, 779, 118, 602, 170, 24, 785, 183, 614, 441,
          571, 621, 693, 706, 470, 608, 752, 25, 685, 381, 806, 174, 824, 196, 286, 330, 737, 833, 503, 633, 474, 91, 265, 831, 184, 788, 422, 751],

         [6, 9, 375, 701, 158, 677, 509, 357, 260, 377, 661, 724, 402, 680, 778, 180, 480, 511, 341, 96, 549, 562, 55, 313, 619, 847, 472, 769, 16,
          691, 18, 432, 296, 342, 455, 409, 811, 781, 94, 247, 545, 726, 165, 603, 224, 530, 630, 128, 148, 376, 458, 830, 428, 679, 704, 401, 418, 290,
          771, 695, 281, 326, 314, 336, 397, 586, 240, 783, 249, 572, 294, 743, 100, 490, 32, 328, 440, 461, 54, 22, 565, 625, 629, 259, 842, 636, 753,
          542, 29, 114, 654, 711, 120, 75, 325, 759, 445, 561, 47, 544, 826, 133, 525, 417, 291, 515, 70, 231, 274, 505, 436, 134, 584, 179, 765, 126, 536,
          69, 707, 814, 488, 838, 65, 777, 316, 368, 426, 430, 594, 624, 506],

         [13, 213, 51, 812, 413, 379, 498, 239, 394, 406, 513, 740, 366, 507, 615, 534, 735, 716, 143, 443, 798, 205, 360,
          611, 671, 33, 627, 245, 365, 147, 532, 209, 161, 405, 324, 256, 650, 805, 332, 504, 763, 35, 226, 23, 244, 427, 447,
          656, 71, 243, 337, 844, 631, 367, 829, 278, 840, 78, 312, 715, 271, 107, 307, 539, 578, 670, 761, 823, 745, 236, 419,
          481, 714, 50, 223, 787, 541, 702, 843, 508, 378, 385, 496, 522, 632, 645, 383, 182, 28, 676, 266, 41, 62, 552, 255, 487,
          122, 334, 303, 105, 157, 227, 739, 178, 822, 372, 190, 280, 262, 827, 343, 710, 215, 784, 390, 361, 815, 555, 452, 659, 129,
          495, 17, 433, 177, 395, 403, 500, 242, 601, 131, 77, 832, 573, 166, 285, 757, 690, 399, 448, 512, 466, 384, 567, 607, 569, 722,
          74, 424, 26, 234, 339, 579, 393, 494, 705, 49, 655, 282, 776, 202]]

        clusters[0] = np.array(sorted(clusters[0]))
        clusters[1] = np.array(sorted(clusters[1]))
        clusters[2] = np.array(sorted(clusters[2]))
        clusters[3] = np.array(sorted(clusters[3]))
        clusters[4] = np.array(sorted(clusters[4]))
        clusters[5] = np.array(sorted(clusters[5]))

        clusters = np.array(clusters)
        print(clusters[0])
        print(len(clusters[0]))
        print("\n")

        print(clusters[1])
        print(len(clusters[1]))
        print("\n")

        print(clusters[2])
        print(len(clusters[2]))
        print("\n")

        print(clusters[3])
        print(len(clusters[3]))
        print("\n")

        print(clusters[4])
        print(len(clusters[4]))
        print("\n")

        print(clusters[5])
        print(len(clusters[5]))
        print("\n")