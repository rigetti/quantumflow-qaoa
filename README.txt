>Notice: This is research code that will not necessarily be maintained to
>support further releases of Forest and other Rigetti Software. We welcome
>bug reports and PRs but make no guarantee about fixes or responses.


# QuantumFlow-QAOA: Optimize QAOA circuits for graph maxcut using tensorflow

TensorFlow open source implementation for training Quantum Approximate
Optimization Algorithm (QAOA) circuits on the graph MaxCut problem, from the 
paper:

[*Performance of the Quantum Approximate Optimization Algorithm
on the Maximum Cut Problem*](https://arxiv.org/abs/XXXX.XXXXX)

by Gavin E. Crooks


## Contact

***Code author:*** Gavin E. Crooks

***Pull requests and issues:*** @gecrooks


## Installation
This code relies upon QuantumFlow: A Quantum Algorithms Development Toolkit

```
git clone https://github.com/rigetticomputing/quantumflow-qaoa.git
cd quantumflow-qaoa
pip install -r requirements.txt
```


## train_qaoa_maxcut_sgd.py

Train a QAOA circuit of N qubits and P steps to find good solutions to the MaxCut 
problem. We train on randomly sampled graphs, and validate against a fixed set
of pregenerated graphs provided by qauntumflow.



```
> ./train_qaoa_maxcut_sgd.py --help
usage: train_qaoa_maxcut_sgd.py [-h] [--version] [-v] [-i FILE] [-o FILE]
                                [-N NODES] [-P STEPS] [--epochs EPOCHS]
                                [--lr LEARNING_RATE] [-T FILE] [-V FILE]

QAOA graph maxcut using tensorflow gradient descent

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -v, --verbose
  -i FILE, --fin FILE   Read model from file
  -o FILE, --fout FILE  Write model to file
  -N NODES, --nodes NODES
  -P STEPS, --steps STEPS
  --epochs EPOCHS
  --lr LEARNING_RATE
  -T FILE, --train FILE
                        Collection of graphs to train on
  -V FILE, --validation FILE
                        Validation graph dataset
 ```

E.g. train 10 epoces on a batch of 100 8 node graphs, with 12 QAOA steps.
```
./train_qaoa_maxcut_sgd.py -N 8 -P 12 --verbose --epochs 10
```


## Citation

If you use this code, please cite our paper:
```
@article{Crooks2018b,
  title={Performance of the Quantum Approximate Optimization Algorithm
on the Maximum Cut Problem},
  author={Crooks, Gavin E},
  note={https://arxiv.org/abs/XXXX.XXXXX},
  year={2018}
}
```
