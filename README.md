Metric Learning of Ranking data
=============

## Requirements
- Python 3: [Anaconda](https://www.anaconda.com/download/) is recommended
- Python libraries: Chainer, chainerui:  `pip install chainer chainerui`
- for parallel learning: mpi4py: `conda install mpi4py`

# How to use
- Create sample arrangements: The following takes the 2D coordinates of points in brands.csv and outputs 
players.csv (containing 100 randomly generated player coordinates) and ranking.csv (ranking inequalities)
```
    python arrangement.py --brand brands.csv -np 100 -g --dim 2
```
Output files are found under `result` directory.
- We can also generate brand coordinates randomly as well:
```
    python arrangement.py -nb 5 -np 100 -g --dim 2
```
- For usage
```
    python arrangement.py -h
```
- Each line in the ranking file ranking.csv has the following format:
```
    pid, x, y
```
where x is closer to the player pid (i.e., x>y for pid).
- Metric Learning: The following takes ranking.csv and outputs out_brands.csv (predicted brand coords) and out_players.csv (predicted player coords) under "~/Downloads/result" directory
```
    python rankLearn.py ranking.csv -e 50 -lr 0.01 -o ~/Downloads/result -ld 10
```
batch size (-b 20), number of epochs (-e 50), learning rate (-lr 0.01), learning rate drop times (-ld 10) have a large impact on the speed and the accuracy of learning.
- Starting from given coordinates (set initial configuration)
```
    python rankLearn.py ranking.csv -e 50 -lr 0.01 -rp 1e-1 -o ~/Downloads/result -ld 10 -p out_players.csv -b out_brands.csv
```
- Parallel learning
```
    mpiexec -n 4 python rankLearn.py ranking.csv -e 50 -lr 0.01 -o ~/Downloads/result --mpi
```
- For usage
```
    python rankLearn.py -h
```
- Verification: ground truth
```
    python arrangement.py --brand brands.csv --player players.csv --plot
```
- Verification: predicted
```
    python arrangement.py --brand out_brands.csv --player out_players.csv --plot
```
