# SpectralClusteringFD
Memory-efficient Spectral Clustering with Frequent Direction

# Special Thanks
- [Scikit-learn](https://github.com/scikit-learn/)
- [hido](https://github.com/hido/frequent-direction)

# Reference
Spectral Clustering
- Normalized cuts and image segmentation, 2000 Jianbo Shi, Jitendra Malik http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324
- A Tutorial on Spectral Clustering, 2007 Ulrike von Luxburg http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323
- Multiclass spectral clustering, 2003 Stella X. Yu, Jianbo Shi http://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf

Frequent Direction
- [Edo. Liberty, "Simple and Deterministic Matrix Sketching", ACM SIGKDD, 2013.](http://www.cs.yale.edu/homes/el327/papers/simpleMatrixSketching.pdf)

# License
BSD 2-clause License

# Install
```
% pip install git+https://github.com/AtsushiHashimoto/frequent_direction.git
% pip install git+https://github.com/AtsushiHashimoto/spectral_clustering_fd.git
```


# Usage
```
import spectral_clustering_fd as scfd

n_clusters = 8 # replace the number along with your problem.
model = scfd.SpectralClusteringFD(n_clusters = n_clusters,
                                      random_state=None,
                                      n_init=10,
                                      gamma=1., affinity='rbf',
                                      assign_labels='discretize',
                                      kernel_params=None, n_jobs=1
                                    )
                                    
X = load_samples(filepath) # replace this code to what you need

labels = model_sketch.fit_predict(X)
```

# Sample Code
```
% wget https://raw.githubusercontent.com/AtsushiHashimoto/spectral_clustering_fd/master/example/sample_spectral_clustering_fd.py
% pip install memory_profiler
% python -m memory_profiler test_sketch.py
```
then,
```
$ python -m memory_profiler sample_spectral_clustering_fd.py 
Labels Ground Truth:  [3 2 3 ..., 3 2 2]
initialization
do fit with sketch SC model...
...done
elapsed_time for fit:7.351032972335815 [sec]
Labels(sketch):  [2 2 2 ..., 2 2 2]
adjusted_mutual_info_score(sketch):  0.48281512646
do fit with original SC model...
...done
elapsed_time for fit:3.453601121902466 [sec]
Labels(orig)  :  [0 1 0 ..., 0 1 1]
adjusted_mutual_info_score(orig)  :  0.718604115552
Filename: sample_spectral_clustering_fd.py

Line #    Mem usage    Increment   Line Contents
================================================
    16   63.887 MiB    0.000 MiB   @profile
    17                             def main():
    18   63.887 MiB    0.000 MiB       n_samples = 2048
    19   63.887 MiB    0.000 MiB       n_features = 16
    20   63.887 MiB    0.000 MiB       n_centers = 4
    21                             
    22   64.699 MiB    0.812 MiB       X, labels_gt = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, cluster_std=1.0, center_box=(-2.0, 2.0), shuffle=True, random_state=None)
    23                             
    24   64.723 MiB    0.023 MiB       print("Labels Ground Truth: ",labels_gt)
    25                             
    26                             
    27   64.723 MiB    0.000 MiB       print("initialization")
    28   64.723 MiB    0.000 MiB       model_sketch = scfd.SpectralClusteringFD(n_clusters = n_centers,
    29   64.723 MiB    0.000 MiB                                         random_state=None,
    30   64.723 MiB    0.000 MiB                                         n_init=10,
    31   64.723 MiB    0.000 MiB                                         gamma=1., affinity='rbf',
    32   64.723 MiB    0.000 MiB                                         assign_labels='discretize',
    33   64.723 MiB    0.000 MiB                                         kernel_params=None, n_jobs=1
    34                                                                 )
    35                             
    36   64.723 MiB    0.000 MiB       print("do fit with sketch SC model...")
    37   64.723 MiB    0.000 MiB       start = time.time()
    38   37.438 MiB  -27.285 MiB       labels_sketch = model_sketch.fit_predict(X)
    39   37.438 MiB    0.000 MiB       elapsed_time = time.time() - start
    40   37.441 MiB    0.004 MiB       print("...done")
    41                             
    42   37.441 MiB    0.000 MiB       print ("elapsed_time for fit:{0}".format(elapsed_time),"[sec]")
    43   37.441 MiB    0.000 MiB       print("Labels(sketch): ",labels_sketch)
    44   37.770 MiB    0.328 MiB       AMUI = sm.adjusted_mutual_info_score(labels_gt,labels_sketch)
    45   37.773 MiB    0.004 MiB       print("adjusted_mutual_info_score(sketch): ",AMUI)
    46                             
    47   37.773 MiB    0.000 MiB       model_orig = SpectralClustering(n_clusters = n_centers,
    48   37.773 MiB    0.000 MiB                                         random_state=None,
    49                             
    50   37.773 MiB    0.000 MiB                                         n_init=10,
    51   37.773 MiB    0.000 MiB                                         gamma=1., affinity='rbf',
    52   37.773 MiB    0.000 MiB                                         assign_labels='discretize',
    53   37.785 MiB    0.012 MiB                                         kernel_params=None, n_jobs=1
    54                                                                 )
    55   37.785 MiB    0.000 MiB       print("do fit with original SC model...")
    56   37.785 MiB    0.000 MiB       start = time.time()
    57  196.258 MiB  158.473 MiB       labels_orig = model_orig.fit_predict(X)
    58  196.258 MiB    0.000 MiB       elapsed_time = time.time() - start
    59  196.258 MiB    0.000 MiB       print("...done")
    60  196.258 MiB    0.000 MiB       print ("elapsed_time for fit:{0}".format(elapsed_time),"[sec]")
    61  196.258 MiB    0.000 MiB       print("Labels(orig)  : ",labels_orig)
    62  196.281 MiB    0.023 MiB       AMUI = sm.adjusted_mutual_info_score(labels_gt,labels_orig)
    63  196.281 MiB    0.000 MiB       print("adjusted_mutual_info_score(orig)  : ",AMUI)
    64                             
    65                                 """
    66                                 ARANDi = sm.adjusted_rand_score(labels_gt,labels)
    67                                 print("adjusted_rand_index: ",ARANDi)
    68                                 MUI = sm.mutual_info_score(labels_gt,labels)
    69                                 print("mutual_info_score: ",MUI)
    70                                 #RANDi = sm.rand_score(data.target,labels)
    71                                 #print("rand_index: ",RANDi)
    72  196.281 MiB    0.000 MiB       """
```

# Summery of Execution Result
Note that the result will be different every execution because of random sample generation.

Exec. time
- SC with fd : 5.571173906326294 [sec] (increase drastically along with the feature dimensions)
- Original SC: 4.347407102584839 [sec]

Adusted Mutual Info. (Very unstable. It highly depends on generated sample set.)
- SC with fd : 0.871839784492
- Original SC: 0.706697461238
 
Memory Consumption
- SC with fd :   1.555 MiB  
- Original SC: 156.418 MiB
