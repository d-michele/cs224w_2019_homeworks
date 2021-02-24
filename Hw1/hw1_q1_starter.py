################################################################################
# CS 224W (Fall 2019) - HW1
# Starter code for Question 1
# Last Updated: Sep 25, 2019
################################################################################

import snap
import numpy as np
import matplotlib.pyplot as plt

# Setup
erdosRenyi = None
smallWorld = None
collabNet = None


def createAdjacencyMatrix(N, E):
    p = 2 * E / (N * (N - 1))
    adj = {}
    for i in range(N):
        adj[i] = {}
    for i in range(N):
        sample = np.random.binomial(size=(N - i - 1), n=1, p=p)
        sample_indexes = np.squeeze(np.argwhere(sample), axis=1)
        sample_indexes = [x + i + 1 for x in sample_indexes]
        adj[i].update({x: 1 for x in sample_indexes})
        for j in adj[i]:
            adj[j][i] = 1

    adj = adjustNumberOfEdges(adj, N, E)

    return adj


def adjustNumberOfEdges(adj, N, E):
    inserted_edges = [len(v) for k, v in adj.items()]
    inserted_edges = sum(inserted_edges) / 2

    delta = E - inserted_edges
    while (delta != 0):
        #     print(delta)
        u = np.random.randint(0, N)
        # insert remaining links
        if (delta > 0):
            v = np.random.randint(0, N)
            if v not in adj[u]:
                adj[u][v] = 1
                adj[v][u] = 1
                delta -= 1
        # delete links
        elif (delta < 0):
            # no links from this node
            if len(adj[u]) == 0:
                continue
            linked_nodes = list(adj[u].keys())
            v = np.random.choice(linked_nodes)
            del adj[u][v]
            del adj[v][u]
            delta += 1
    return adj


# Problem 1.1
def genErdosRenyi(N=5242, E=14484):
    """
    :param - N: number of nodes
    :param - E: number of edges

    return type: snap.PUNGraph
    return: Erdos-Renyi graph with N nodes and E edges
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.TUNGraph().New()
    for i in range(N):
        Graph.AddNode(i)

    adj = createAdjacencyMatrix(N, E)
    for k1, v1 in adj.items():
        for k2, v2 in adj[k1].items():
            Graph.AddEdge(int(k1), int(k2))

    ############################################################################
    return Graph


def genCircle(N=5242):
    """
    :param - N: number of nodes

    return type: snap.PUNGraph
    return: Circle graph with N nodes and N edges. Imagine the nodes form a
        circle and each node is connected to its two direct neighbors.
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.TUNGraph().New()
    for i in range(N):
        Graph.AddNode(i)

    for u in range(N):
        Graph.AddEdge(u, (u + 1) % N)

    ############################################################################
    return Graph


def connectNbrOfNbr(Graph, N=5242):
    """
    :param - Graph: snap.PUNGraph object representing a circle graph on N nodes
    :param - N: number of nodes

    return type: snap.PUNGraph
    return: Graph object with additional N edges added by connecting each node
        to the neighbors of its neighbors
    """
    ############################################################################
    # TODO: Your code here!
    edgeList = []
    for NI in Graph.Nodes():
        for Id in NI.GetOutEdges():
            for IdN in Graph.GetNI(Id).GetOutEdges():
                if NI.GetId() == IdN:
                    continue
                edgeList.append((NI.GetId(), IdN))

    for edge in edgeList:
        Graph.AddEdge(*edge)
    ############################################################################
    return Graph


def connectRandomNodes(Graph, M=4000):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph
    :param - M: number of edges to be added

    return type: snap.PUNGraph
    return: Graph object with additional M edges added by connecting M randomly
        selected pairs of nodes not already connected.
    """
    ############################################################################
    # TODO: Your code here!
    N = Graph.GetNodes()
    toInsert = M
    while (toInsert > 0):
        u = np.random.randint(0, N)
        # insert remaining links
        v = np.random.randint(0, N)
        if not Graph.IsEdge(u, v):
            Graph.AddEdge(u, v)
            toInsert -= 1

    ############################################################################
    return Graph


def genSmallWorld(N=5242, E=14484):
    """
    :param - N: number of nodes
    :param - E: number of edges

    return type: snap.PUNGraph
    return: Small-World graph with N nodes and E edges
    """
    Graph = genCircle(N)
    Graph = connectNbrOfNbr(Graph, N)
    Graph = connectRandomNodes(Graph, 4000)
    return Graph


def loadCollabNet(path):
    """
    :param - path: path to edge list file

    return type: snap.PUNGraph
    return: Graph loaded from edge list at `path and self edges removed

    Do not forget to remove the self edges!
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.LoadEdgeList(snap.TUNGraph, path)
    # delete self edfges
    for NI in Graph.Nodes():
        Graph.DelEdge(NI.GetId(), NI.GetId())

    ############################################################################
    return Graph


def getDataPointsToPlot(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return values:
    X: list of degrees
    Y: list of frequencies: Y[i] = fraction of nodes with degree X[i]
    """
    ############################################################################
    # TODO: Your code here!
    X, Y = [], []
    for item in Graph.GetDegCnt():
        X.append(item.GetVal1())
        Y.append(item.GetVal2())
    # nodedegree = {}

    # for NI in Graph.Nodes():
    # k = NI.GetDeg
    # X.append(k)
    # Y[X[k]] = 1 if Y[X[k]] == None else Y[X[i]] + 1

    ############################################################################
    return X, Y


def Q1_1():
    """
    Code for HW1 Q1.1
    """
    global erdosRenyi, smallWorld, collabNet
    erdosRenyi = genErdosRenyi(5242, 14484)
    smallWorld = genSmallWorld(5242, 14484)
    collabNet = loadCollabNet("data/ca-GrQc.txt")

    x_erdosRenyi, y_erdosRenyi = getDataPointsToPlot(erdosRenyi)
    plt.loglog(x_erdosRenyi, y_erdosRenyi, color='y', label='Erdos Renyi Network')

    x_smallWorld, y_smallWorld = getDataPointsToPlot(smallWorld)
    plt.loglog(
        x_smallWorld,
        y_smallWorld,
        linestyle='dashed',
        color='r',
        label='Small World Network')

    x_collabNet, y_collabNet = getDataPointsToPlot(collabNet)
    plt.loglog(
        x_collabNet,
        y_collabNet,
        linestyle='dotted',
        color='b',
        label='Collaboration Network')

    plt.xlabel('Node Degree (log)')
    plt.ylabel('Proportion of Nodes with a Given Degree (log)')
    plt.title(
        'Degree Distribution of Erdos Renyi, Small World, and Collaboration Networks')
    plt.legend()
    plt.show()


# Execute code for Q1.1
Q1_1()

# Problem 1.2 - Clustering Coefficient


def calcClusteringCoefficientSingleNode(Node, Graph):
    """
    :param - Node: node from snap.PUNGraph object. Graph.Nodes() will give an
                   iterable of nodes in a graph
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: float
    returns: local clustering coeffient of Node
    """
    ############################################################################
    # TODO: Your code here!
    k = Node.GetDeg()
    if k < 2:
        return 0.0

    e = 0
    for i in range(k):
        neigh_i = Node.GetNbrNId(i)
        for j in range(i + 1, k):
            neigh_j = Node.GetNbrNId(j)
            neighI_j = Graph.GetNI(neigh_j)
            e += neighI_j.IsNbrNId(neigh_i)

    C = 2 * e / (k * (k - 1))
    ############################################################################
    return C


def calcClusteringCoefficient(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: float
    returns: clustering coeffient of Graph
    """
    ############################################################################
    # TODO: Your code here! If you filled out calcClusteringCoefficientSingleNode,
    #       you'll probably want to call it in a loop here
    C = 0.0
    V = Graph.GetNodes()
    for NI in Graph.Nodes():
        C += calcClusteringCoefficientSingleNode(NI, Graph)
    C = C / V
    ############################################################################
    return C


def Q1_2():
    """
    Code for Q1.2
    """
    C_erdosRenyi = calcClusteringCoefficient(erdosRenyi)
    C_smallWorld = calcClusteringCoefficient(smallWorld)
    C_collabNet = calcClusteringCoefficient(collabNet)

    print('Clustering Coefficient for Erdos Renyi Network: %f' % C_erdosRenyi)
    print('Clustering Coefficient for Small World Network: %f' % C_smallWorld)
    print('Clustering Coefficient for Collaboration Network: %f' % C_collabNet)


# Execute code for Q1.2
Q1_2()
