# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import time

import spectral_clustering_fd as scfd

from sklearn.cluster import SpectralClustering

from sklearn.datasets import make_blobs
import random
import sklearn.metrics as sm

@profile
def main():
    n_samples = 2048
    n_features = 16
    n_centers = 4

    X, labels_gt = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, cluster_std=1.0, center_box=(-2.0, 2.0), shuffle=True, random_state=None)

    print("Labels Ground Truth: ",labels_gt)


    print("initialization")
    model_sketch = scfd.SpectralClusteringFD(n_clusters = n_centers,
                                      random_state=None,
                                      n_init=10,
                                      normed=False,
                                      gamma=1., affinity='rbf',
                                      assign_labels='discretize',
                                      kernel_params=None, n_jobs=1
                                    )

    print("do fit with sketch SC model...")
    start = time.time()
    labels_sketch = model_sketch.fit_predict(X)
    elapsed_time = time.time() - start
    print("...done")

    print ("elapsed_time for fit:{0}".format(elapsed_time),"[sec]")
    print("Labels(sketch): ",labels_sketch)
    AMUI = sm.adjusted_mutual_info_score(labels_gt,labels_sketch)
    print("adjusted_mutual_info_score(sketch): ",AMUI)

    model_orig = SpectralClustering(n_clusters = n_centers,
                                      random_state=None,

                                      n_init=10,
                                      gamma=1., affinity='rbf',
                                      assign_labels='discretize',
                                      kernel_params=None, n_jobs=1
                                    )
    print("do fit with original SC model...")
    start = time.time()
    labels_orig = model_orig.fit_predict(X)
    elapsed_time = time.time() - start
    print("...done")
    print ("elapsed_time for fit:{0}".format(elapsed_time),"[sec]")
    print("Labels(orig)  : ",labels_orig)
    AMUI = sm.adjusted_mutual_info_score(labels_gt,labels_orig)
    print("adjusted_mutual_info_score(orig)  : ",AMUI)

    """
    ARANDi = sm.adjusted_rand_score(labels_gt,labels)
    print("adjusted_rand_index: ",ARANDi)
    MUI = sm.mutual_info_score(labels_gt,labels)
    print("mutual_info_score: ",MUI)
    #RANDi = sm.rand_score(data.target,labels)
    #print("rand_index: ",RANDi)
    """

if __name__ == "__main__":
    main()
