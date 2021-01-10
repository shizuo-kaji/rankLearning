Learning ranking distributions using hyperplane arrangement
=============
Written by Shizuo KAJI

Look at the following paper for details:
- A hyper-surface arrangement model of ranking distributions by S. Kaji, A. Horiguchi, T. Abe, and Y. Watanabe.

# Requirements
- Python 3: [Anaconda](https://www.anaconda.com/download/) is recommended
- Python libraries: Chainer, chainerui:  `pip install chainer chainerui`
- (optional) for parallel learning: mpi4py: `conda install mpi4py`

# Terminology
- label = alternative = item are those that are ranked. 
- instance = agent = judge are those who rank labels.
- ranking is a partial ordering of labels by an instance.
- ranking distribution is a probability distribution on the ordering of labels which occurs from a collection of rankings.
- In our model, labels and instance are represented by points in the m-dimensional ball.
- ranking distribution can be represented by a n!-dimensional vector, where n is the number of labels.
- Our model approximates the ranking distribution by the distribution of the volumes of the cells formed by the hyper-plane arrangements in the m-dimensional ball. The dimension of our model is nm.

# How to use
- Partial ranking data should be prepared in a csv file in which each line has the following format:
```
    id, x, y, z, ...
```
where x > y > z > ... for instance id. There can be multiple lines having the same id so that any partial ordering can be specified.

- Learning the model: The following learns the ranking distribution of ranking.csv and
outputs `labels.csv` (learned label coords) and `instances.csv` (learned instance coords) under the directory named `result`.
```
    python rankLearn.py ranking.csv -e 50 -lr 0.01 -lri 1e-3 -o result
```
The batch size (`-b 20`), number of epochs (`-e 50`), learning rate (`-lr 0.01`) have a large impact on the speed and the accuracy of learning.
Note that only labels.csv is needed for sampling from the leaned model.

- Learning can be resumed by giving initial configurations of label and/or instance coordinates.
```
    python rankLearn.py ranking.csv -e 50 -lr 0.01 -lri 1e-3 -o result -i instances.csv -l labels.csv
```

- Parallel learning: if mpi4py is installed, learning can be parallelised using MPI.
```
    mpiexec -n 4 python rankLearn.py ranking.csv -e 50 -lr 0.01 -o result --mpi
```

- To see the list of command-line options, 
```
    python rankLearn.py -h
```


# Evaluating the model
- To compare the learned model with the data in terms of various metrics,
```
    python arrangement.py --label labels.csv -r2 ranking.csv --compute_wasserstein --top_n 3
```

- To plot the coordinates of labels and instances,
```
    python arrangement.py --label labels.csv --instance instances.csv --plot
```

# Sample ranking data creation
- To sample 1000 rankings from the learned model `labels.csv`, 
```
    python arrangement.py --label labels.csv -ni 1000 -g --dim 2 -o result
```
The sampled rankings (`ranking.csv`) and instance coordinates (`instances.csv`) are found under the directory named `result`.
- To create a 2D sample arrangement with 5 labels and 1000 instances,
```
    python arrangement.py -nl 5 -ni 1000 -g --dim 2
```
- To see the list of command-line options,
```
    python arrangement.py -h
```


