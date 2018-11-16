#!/usr/bin/env python

# Copyright 2018, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Optimize QAOA circuits for graph maxcut, using tensorflow gradient descent.

Train on batches of graphs, not one graph at a time.
"""

# DISCLAIMER: Research grade coding.

import argparse
import json
import sys
import os

import numpy as np

import networkx as nx
import tensorflow as tf

# Note that we QuantumFlow sets an interactive session
os.environ['QUANTUMFLOW_BACKEND'] = 'tensorflow'
import quantumflow as qf                                # noqa
assert qf.backend.BACKEND == 'tensorflow'

from quantumflow.qaoa import qubo_circuit, graph_cuts   # noqa


__version__ = qf.__version__
__description__ = 'QAOA graph maxcut using tensorflow gradient descent'

TRAINING_GRAPHS = 100
VAL_GRAPHS = 100

DEFAULT_NODES = 6
DEFAULT_STEPS = 8
LEARNING_RATE = 0.01
EPOCHS = 10
MAX_OPT_STEPS = 10000
MIN_DIFF = 0.0001
INIT_SCALE = 0.01
INIT_BIAS = 0.5


class QAOAMaxcutModel:
    def __init__(self,
                 nodes: int,
                 steps: int,
                 init_beta=None,
                 init_gamma=None,
                 init_scale: float = None) -> None:
        """Build the tensorflow computational graph for QAOA MAXCUT"""

        if init_scale is None:
            init_scale = INIT_SCALE

        # Initialize the parameters that need to be learned
        if init_beta is None:
            init_beta = np.random.normal(loc=INIT_BIAS, scale=init_scale,
                                         size=[steps])
        beta = tf.get_variable('beta',
                               initializer=init_beta, dtype=tf.float64)

        if init_gamma is None:
            init_gamma = np.random.normal(loc=INIT_BIAS, scale=init_scale,
                                          size=[steps])
        gamma = tf.get_variable('gamma',
                                initializer=init_gamma, dtype=tf.float64)

        # Create placeholders for graph adjacency matrix and cut Hamiltonian
        edge_weights = tf.placeholder(tf.float64,
                                      shape=[nodes, nodes],
                                      name='edge_weights')
        hamiltonian = tf.placeholder(tf.complex128,
                                     shape=[2]*nodes,
                                     name='hamiltonian')

        # Create generic MAXCUT graph
        weighted_graph = nx.Graph()
        for n0 in range(nodes):
            for n1 in range(nodes):
                if n0 > n1:
                    weighted_graph.add_edge(n0, n1,
                                            weight=edge_weights[n0, n1])

        # Create MAXCUT circuit
        circ = qubo_circuit(weighted_graph, steps, beta, gamma)

        # Create tensorflow computational graph
        ket = circ.run()
        mean_cut = ket.expectation(hamiltonian)

        self.beta = beta
        self.gamma = gamma
        self.edge_weights = edge_weights
        self.hamiltonian = hamiltonian
        self.mean_cut = mean_cut
        self.ket = ket

    def dump(self, fout):
        sess = tf.get_default_session()
        beta = sess.run(self.beta)
        gamma = sess.run(self.gamma)
        data = {'beta': list(beta), 'gamma': list(gamma)}
        json.dump(data, fout)

    def _feed(self, graph):
        graph_adjacency = nx.to_numpy_matrix(graph).astype(dtype=np.double)
        graph_hamiltonian = graph_cuts(graph)

        feed_dict = {self.edge_weights: graph_adjacency,
                     self.hamiltonian: graph_hamiltonian}

        return feed_dict

    def train(self, graphs,
              validation=None,
              epochs=1,
              learning_rate=LEARNING_RATE,
              verbose=False):

        if verbose:
            print("# [QAOA MaxCut TF train] epochs:{} lr:{}".format(epochs,
                  learning_rate))

        G = len(graphs)

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = opt.minimize(-self.mean_cut, var_list=[self.beta, self.gamma])

        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())

        for nepoch in range(epochs):
            ratio_total = 0

            for g, graph in enumerate(graphs):
                feed = self._feed(graph)
                train.run(feed_dict=feed)

                if verbose:
                    maxcut = feed[self.hamiltonian].max()
                    mean_cut = sess.run(self.mean_cut, feed_dict=feed)
                    ratio = mean_cut / maxcut
                    ratio_total += ratio

                    sys.stdout.write(
                        '\repoch: {} graph: {}/{} mean_cut: {:<06.4}'.format(
                            nepoch, g, G, ratio))

            if validation is not None:
                val_ratio = 0
                for g, graph in enumerate(validation):
                    feed = self._feed(graph)
                    maxcut = feed[self.hamiltonian].max()
                    mean_cut = sess.run(self.mean_cut, feed_dict=feed)
                    ratio = mean_cut / maxcut
                    val_ratio += ratio
                val_ratio /= len(validation)

                if verbose:
                    sys.stdout.write(' val_cut: {:<06.4}'.format(val_ratio))

            if verbose:
                print()


# ---------- Command Line Interface ----------
def _cli():

    parser = argparse.ArgumentParser(
        description=__description__)

    parser.add_argument('--version', action='version', version=__version__)

    parser.add_argument('-v', '--verbose', action='store_true')

    parser.add_argument('-i', '--fin', action='store', dest='fin',
                        default='', metavar='FILE',
                        help='Read model from file')

    parser.add_argument('-o', '--fout', action='store', dest='fout',
                        default='', metavar='FILE',
                        help='Write model to file')

    parser.add_argument('-N', '--nodes', type=int, dest='nodes',
                        default=DEFAULT_NODES)

    parser.add_argument('-P', '--steps', type=int, dest='steps',
                        default=DEFAULT_STEPS)

    parser.add_argument('--epochs', type=int, dest='epochs',
                        default=EPOCHS)

    parser.add_argument('--lr', type=float, dest='learning_rate',
                        default=LEARNING_RATE)

    parser.add_argument('-T', '--train', action='store', dest='ftrain',
                        default='', metavar='FILE',
                        help='Collection of graphs to train on')

    parser.add_argument('-V', '--validation',
                        action='store', dest='fvalidation',
                        default='', metavar='FILE',
                        help='Validation graph dataset')

    opts = vars(parser.parse_args())

    verbose = opts.pop('verbose')
    epochs = opts.pop('epochs')
    steps = opts.pop('steps')
    nodes = opts.pop('nodes')
    learning_rate = opts.pop('learning_rate')

    fin = opts.pop('fin')
    fout = opts.pop('fout')

    ftrain = opts.pop('ftrain')
    fvalidation = opts.pop('fvalidation')

    if ftrain:
        graphs = nx.read_graph6(ftrain)
    else:
        graphs = [nx.gnp_random_graph(nodes, 0.5)
                  for _ in range(TRAINING_GRAPHS)]

    if fvalidation:
        validation = nx.read_graph6(fvalidation)
    else:
        validation = qf.datasets.load_stdgraphs(nodes)

    init_beta = None
    init_gamma = None
    if fin:
        with open(fin) as f:
            data = json.load(f)
        init_beta = np.asarray(data['beta'])
        init_gamma = np.asarray(data['gamma'])

    model = QAOAMaxcutModel(nodes, steps, init_beta, init_gamma)
    model.train(graphs, validation, epochs, learning_rate, verbose)

    if fout:
        with open(fout, 'w') as f:
            model.dump(f)


if __name__ == "__main__":
    _cli()
