import numpy as np
import networkx as nx
from numpy.core.defchararray import multiply
from numpy.lib.index_tricks import AxisConcatenator


def Q1():
    G = generate_graph(debug=True)
    compute_beliefs(G)


def generate_graph(debug=False):
    G = nx.Graph()
    psi_12 = np.array([[1.0, 0.9], [0.9, 1.0]])
    psi_34 = np.array([[1.0, 0.9], [0.9, 1.0]])
    psi_23 = np.array([[0.1, 1.0], [1.0, 0.1]])
    psi_35 = np.array([[0.1, 1.0], [1.0, 0.1]])
    phi_2 = np.array([[1.0, 0.1], [0.1, 1.0]])
    phi_4 = np.array([[1.0, 0.1], [0.1, 1.0]])
    G.add_edge('x1', 'x2', p=psi_12)
    G.add_edge('x2', 'y2', p=phi_2)
    G.add_edge('x2', 'x3', p=psi_23)
    G.add_edge('x3', 'x4', p=psi_34)
    G.add_edge('x3', 'x5', p=psi_35)
    G.add_edge('x4', 'y4', p=phi_4)
    if debug:
        nx.draw(G, node_size=800, with_labels=True, alpha=0.6)
    return G


def compute_beliefs(G):
    # m4_3 = G['x3']['x4']['p'] @ (G['x4']['y4']['p'][:, 1])
    # m5_3 = G['x3']['x5']['p']
    # print(m4_3.shape)
    # print(m5_3.shape)
    # m3_2 = G['x2']['x3']['p'] * m4_3 @ m5_3
    # m2_1 = G['x1']['x2']['p'] @ (G['x2']['y2']['p'][:, 0]) @ (m3_2)
    # b1 = m2_1

    # Dot product since we multiply by the prior and aggregate on x4
    # Sum and elementwise multiplication can be substituted with dot in some cases
    # product but we prefear a better readability in this simple computation.
    # Note that all the mx_y are horizontal vector since in the matrix the information
    # about the variable on which change state as we move in the column, the second dimension
    m4_3 = G['x3']['x4']['p'] @ (G['x4']['y4']['p'][:, 1])
    m5_3 = np.sum(G['x3']['x5']['p'], axis=1)
    m3_2 = np.sum(G['x2']['x3']['p'] * m4_3 * m5_3, axis=1)
    m2_1 = np.sum(G['x1']['x2']['p'] * G['x2']['y2']['p'][:, 0] * m3_2, axis=1)
    b1 = m2_1
    m1_2 = np.sum(G['x1']['x2']['p'], axis=0)
    b2 = G['x2']['y2']['p'][:, 0] * m3_2 * m1_2
    m2_3 = np.sum(G['x2']['x3']['p'].T * G['x2']['y2']['p'][:, 0] * m1_2,
                  axis=1)
    b3 = m2_3 * m4_3 * m5_3
    m3_4 = np.sum(G['x3']['x4']['p'].T * m2_3, axis=1)
    b4 = G['x4']['y4']['p'][:, 1] * m3_4
    m3_5 = np.sum(G['x3']['x5']['p'].T * m2_3, axis=1)
    b5 = m3_5

    print(f'b1: {b1}')
    print(f'b2: {b2}')
    print(f'b3: {b3}')
    print(f'b4: {b4}')
    print(f'b5: {b5}')