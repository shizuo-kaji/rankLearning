Metric Learning of Ranking data
=============
Written by Shizuo KAJI

## Requirements
- Python 3: [Anaconda](https://www.anaconda.com/download/) is recommended
- Python libraries: Chainer, chainerui:  `pip install chainer chainerui`
- for parallel learning: mpi4py: `conda install mpi4py`

# How to use
- Create sample arrangements: The following takes the 2D coordinates of points in labels.csv and outputs 
instances.csv (containing 100 randomly generated instance coordinates) and ranking.csv (ranking inequalities)
```
    python arrangement.py --label labels.csv -np 100 -g --dim 2
```
Output files are found under `result` directory.
- We can also generate label coordinates randomly as well:
```
    python arrangement.py -nb 5 -np 100 -g --dim 2
```
- For usage
```
    python arrangement.py -h
```
- Each line in the ranking file ranking.csv has the following format:
```
    pid, x, y, z, ...
```
where x > y > z > ... for instance pid.

- Metric Learning: The following takes ranking.csv and outputs out_labels.csv (predicted label coords) and out_instances.csv (predicted instance coords) under "~/Downloads/result" directory
```
    python rankLearn.py ranking.csv -e 50 -lr 0.01 -lri 1e-3 -o ~/result -ld 3
```
batch size (-b 20), number of epochs (-e 50), learning rate (-lr 0.01) have a large impact on the speed and the accuracy of learning.

- Starting from given coordinates (set initial configuration)
```
    python rankLearn.py ranking.csv -e 50 -lr 0.01 -lri 1e-3 -o ~/result -ld 2 -p out_instances.csv -b out_labels.csv
```

- Parallel learning
```
    mpiexec -n 4 python rankLearn.py ranking.csv -e 50 -lr 0.01 -o ~/result --mpi
```

- For usage
```
    python rankLearn.py -h
```

- Verification: ground truth
```
    python arrangement.py --label labels.csv --instance instances.csv --plot
```

- Verification: predicted
```
    python arrangement.py --label out_labels.csv --instance out_instances.csv --plot
```
