from typing import Collection
from networkx.classes.function import edges, neighbors, path_weight
from numpy.core.function_base import add_newdoc
import snap
import numpy as np
import heapq
import matplotlib.pyplot as plt
import networkx as nx


def load_netscience_graph(path='hw1-q2.graph'):
    Graph = snap.TUNGraph.Load(snap.TFIn(path))
    return Graph


def get_basic_features(node_id, graph):

    k = graph.GetNI(node_id).GetDeg()
    num_intra_edges, num_inter_edges = get_egonet_inter_intra_num_edges(
        node_id, graph)
    # num_intra_edges = egonet.GetEdges()

    return np.array([k, num_intra_edges, num_inter_edges])


def get_recursive_features(graph, K=2):
    features_matrix = np.zeros((graph.GetNodes(), 3))
    for v in graph.Nodes():
        id = v.GetId()
        features_matrix[id] = get_basic_features(id, graph)
    for i in range(K):
        sum_feat = np.zeros(features_matrix.shape)
        mean_feat = np.zeros(features_matrix.shape)
        for node in graph.Nodes():
            neigh_size = node.GetDeg()
            id = node.GetId()
            for neigh_id in [node.GetNbrNId(i) for i in range(neigh_size)]:
                sum_feat[id] += features_matrix[neigh_id]
                mean_feat[id] = sum_feat[id] / neigh_size if neigh_size != 0 \
                    else mean_feat

        features_matrix = np.concatenate(
            [features_matrix, mean_feat, sum_feat], axis=1)

    return features_matrix


def get_egonet_inter_intra_num_edges(node_id, graph):
    NI = graph.GetNI(node_id)
    num_inter_edges = 0
    num_intra_edges = 0
    k = NI.GetDeg()
    for i in range(k):
        neigh_i_id = NI.GetNbrNId(i)
        NI_i = graph.GetNI(neigh_i_id)
        i_intra_edges = 0
        for j in range(i + 1, k):
            neigh_j_id = NI.GetNbrNId(j)
            i_intra_edges += int(NI_i.IsNbrNId(neigh_j_id))

        # we need to consider the intra node counted twice since we jump
        # some by taking only j greater than i
        # we subtract 1 for the link with the ego node
        num_inter_edges += NI_i.GetDeg() - 2 * i_intra_edges - 1
        # plus edge with ego
        num_intra_edges += i_intra_edges + 1

    return num_intra_edges, num_inter_edges


def cosine_similarity(u: np.array, v: np.array):

    u_sq = np.linalg.norm(u)
    v_sq = np.linalg.norm(v)
    if u_sq == 0 or v_sq == 0:
        sim = np.array(0.0)
    else:
        sim = u.dot(v) / (u_sq * v_sq)

    return sim[()]


def topk_most_similar(node_id, graph, k=5):
    topk = []
    for NI in graph.Nodes():
        id = NI.GetId()
        if id == node_id:
            continue
        u = get_basic_features(node_id, graph)
        v = get_basic_features(id, graph)
        sim = cosine_similarity(u, v)
        if len(topk) < k or sim > topk[0][0]:
            if len(topk) == k:
                heapq.heappop(topk)
            heapq.heappush(topk, (sim, id))

    return sorted(topk, key=lambda x: x[0], reverse=True)


def topk_most_similar_feature_matrix(node_id, features_matrix, k=5):
    topk = []
    u = features_matrix[node_id]
    for i in range(features_matrix.shape[0]):
        if i == node_id:
            continue
        v = features_matrix[i]
        sim = cosine_similarity(u, v)
        if len(topk) < k or sim > topk[0][0]:
            if len(topk) == k:
                heapq.heappop(topk)
            heapq.heappush(topk, (sim, i))

    return sorted(topk, key=lambda x: x[0], reverse=True)


def compute_all_similarities_with_node(node_id, features_matrix):
    x = np.array([])
    u = features_matrix[node_id]
    for i in range(features_matrix.shape[0]):
        v = features_matrix[i]
        sim = cosine_similarity(u, v)
        x = np.append(x, sim)

    return x


def plot_similarity_distribution(node_id, features_matrix):
    # y = np.array()
    x = compute_all_similarities_with_node(node_id, features_matrix)
    n, bins, patches = plt.hist(x, 20, facecolor='y', alpha=0.70)
    plt.xlabel('Similarity')
    plt.ylabel('Occurrence')
    plt.show()

    return n, bins, patches, x


def digitize_similarities(features_matrix,
                          node_id=None,
                          similarities_vector=None,
                          bins=None,
                          numbins=20):
    if similarities_vector == None and node_id == None:
        raise ValueError(
            "Either node_id or similarities_vector must be provided")
    if similarities_vector == None:
        similarities_vector = compute_all_similarities_with_node(
            node_id, features_matrix)
    if bins == None:
        bins = np.histogram_bin_edges(similarities_vector, bins=numbins)

    bin_assignment = np.digitize(similarities_vector, bins)
    bin_dict = {
        b: np.where(bin_assignment == b + 1)[0]
        for b in range(0, numbins)
    }
    #include elements equal to 1, typically the node itself in the last bin
    bin_dict[numbins - 1] = np.concatenate(
        [bin_dict[numbins - 1], (np.where(bin_assignment == numbins + 1)[0])])
    return bin_dict, bins


def draw_subgraph_node(node_id, graph, depth=3):
    edges = get_edges_BFS(node_id, depth, graph)
    subgraph = nx.Graph()
    if len(edges) == 0:
        subgraph.add_node(node_id)
    subgraph.add_edges_from(edges)
    nodes = list(subgraph.nodes())
    node_color = {n: 'b' if n != node_id else 'r' for n in nodes}
    node_color
    nx.draw(subgraph,
            nodelist=node_color.keys(),
            node_size=1000,
            node_color=node_color.values(),
            with_labels=True,
            alpha=0.6)
    plt.show()


def get_edges_BFS(node_id, max_depth, graph):
    visited = []
    queue = []
    edges = []
    visited.append(node_id)
    queue.append((node_id, 0))

    while queue:
        node_id, depth = queue.pop(0)
        NI = graph.GetNI(node_id)
        k = NI.GetDeg()
        if (depth < max_depth):
            for i in range(k):
                neigh_id = NI.GetNbrNId(i)
                edges.append((node_id, neigh_id))
                if neigh_id not in visited:
                    visited.append(neigh_id)
                    queue.append((neigh_id, depth + 1))
        else:
            # capture edges with already discovered nodes
            neighbors = [NI.GetNbrNId(i) for i in range(k)]
            already_discovered_neigh = filter(lambda x: x in visited,
                                              neighbors)
            for neigh in already_discovered_neigh:
                edges.append((node_id, neigh))

    return edges
