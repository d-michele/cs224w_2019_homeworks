import snap
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def load_graph(name):
    '''
    Helper function to load graphs.
    Use "epinions" for Epinions graph and "email" for Email graph.
    Check that the respective .txt files are in the same folder as this script;
    if not, change the paths below as required.
    '''
    if name == "epinions":
        G = snap.LoadEdgeList(snap.PNGraph, "soc-Epinions1.txt", 0, 1)
    elif name == 'email':
        G = snap.LoadEdgeList(snap.PNGraph, "email-EuAll.txt", 0, 1)
    else:
        raise ValueError("Invalid graph: please use 'email' or 'epinions'.")
    return G


def q1_1():
    '''
    You will have to run the inward and outward BFS trees for the 
    respective nodes and reason about whether they are in SCC, IN or OUT.
    You may find the SNAP function GetBfsTree() to be useful here.
    '''

    ##########################################################################
    # TODO: Run outward and inward BFS trees from node 2018, compare sizes
    # and comment on where node 2018 lies.
    G = load_graph("email")
    # Your code here:
    node_id = 2018
    out_disc, in_disc, scc, = node_position(G, node_id)
    print(f'Node: {node_id}')
    print(f'|In({node_id})|: {len(in_disc)}')
    print(f'|Out({node_id})|: {len(out_disc)}')
    print(f'|SCC({node_id})|: {len(scc)}')

    ##########################################################################

    ##########################################################################
    # TODO: Run outward and inward BFS trees from node 224, compare sizes
    # and comment on where node 224 lies.
    G = load_graph("epinions")
    # Your code here:
    node_id = 224
    out_disc, in_disc, scc, = node_position(G, node_id)
    print(f'Node: {node_id}')
    print(f'|In({node_id})|: {len(in_disc)}')
    print(f'|Out({node_id})|: {len(out_disc)}')
    print(f'|SCC({node_id})|: {len(scc)}')
    ##########################################################################

    print('2.1: Done!\n')


def node_position(G: snap.PNGraph, node_id):
    outward_discovered = bfs(G, node_id)
    inward_discovered = bfs(G, node_id, out_direction=False)
    scc = inward_discovered.intersection(outward_discovered)
    return inward_discovered, outward_discovered, scc


def q1_2():
    '''
    For each graph, get 100 random nodes and find the number of nodes in their
    inward and outward BFS trees starting from each node. Plot the cumulative
    number of nodes reached in the BFS runs, similar to the graph shown in
    Broder et al. (see Figure in handout). You will need to have 4 figures,
    one each for the inward and outward BFS for each of email and epinions.

    Note: You may find the SNAP function GetRndNId() useful to get random
    node IDs (for initializing BFS).
    '''
    ##########################################################################
    # TODO: See above.
    # Your code here:
    g = load_graph('email')
    plot_in_out_reachability(g, network_name='Email')
    g = load_graph('epinions')
    plot_in_out_reachability(g, network_name='Epinions')
    ##########################################################################
    print('2.2: Done!\n')


def plot_in_out_reachability(g: snap.PNGraph, random_size: int = 100, network_name: str = ''):
    random_nodes = np.random.randint(low=0, high=g.GetNodes(), size=100).tolist()
    out_reachability = []
    in_reachability = []
    for v in random_nodes:
        out_reachability.append(g.GetBfsTree(v, True, False).GetNodes())
        in_reachability.append(g.GetBfsTree(v, False, True).GetNodes())
    in_reachability.sort()
    out_reachability.sort()
    x = np.linspace(0, 1, num=random_size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(network_name)
    ax1.plot(x, out_reachability, 'k')
    ax1.set(ylabel='# nodes reached')
    ax1.set(xlabel='Reachability using outlinks')
    ax1.set_yscale('log')
    ax2.plot(x, in_reachability, 'k')
    ax2.set(ylabel='# nodes reached', yscale='log')
    ax2.set(xlabel='Reachability using inlinks')
    ax2.set_yscale('log')
    plt.show()

    return out_reachability, in_reachability


def q1_3():
    '''
    For each graph, determine the size of the following regions:
        DISCONNECTED
        IN
        OUT
        SCC
        TENDRILS + TUBES
You can use SNAP functions GetMxWcc() and GetMxScc() to get the sizes of the largest WCC and SCC on each graph.
    '''
    ##########################################################################
    # TODO: See above.
    # Your code here:
    g = load_graph('email')
    disc, ins, out, scc, tendril_tubes = get_regions_sets(g)

    print(f'Disconnected:{len(disc)}')
    print(f'In:{len(ins)}')
    print(f'Out:{len(out)}')
    print(f'SCC:{len(scc)}')
    print(f'Tendrils + tubes:{len(tendril_tubes)}')

    g = load_graph('epinions')
    disc, ins, out, scc, tendril_tubes = get_regions_sets(g)

    print(f'Disconnected:{len(disc)}')
    print(f'In:{len(ins)}')
    print(f'Out:{len(out)}')
    print(f'SCC:{len(scc)}')
    print(f'Tendrils + tubes:{len(tendril_tubes)}')
    ##########################################################################
    print('2.3: Done!\n')


def get_regions_sets(g: snap.PNGraph):
    nodes = {v.GetId() for v in g.Nodes()}
    wcc = {v.GetId() for v in g.GetMxWcc().Nodes()}
    disc = nodes.difference(wcc)
    scc = {v.GetId() for v in g.GetMxScc().Nodes()}
    v = next(iter(scc))
    in_scc = {v.GetId() for v in g.GetBfsTree(v, False, True).Nodes()}
    out_scc = {v.GetId() for v in g.GetBfsTree(v, True, False).Nodes()}
    out = out_scc.difference(scc)
    ins = in_scc.difference(scc)
    tendril_tubes = wcc.difference(ins).difference(out).difference(scc)

    return disc, ins, out, scc, tendril_tubes


def q1_4():
    '''
    For each graph, calculate the probability that a path exists between
    two nodes chosen uniformly from the overall graph.
    You can do this by choosing a large number of pairs of random nodes
    and calculating the fraction of these pairs which are connected.
    The following SNAP functions may be of help: GetRndNId(), GetShortPath()
    '''
    ##########################################################################
    # TODO: See above.
    # Your code here:
    g = load_graph('email')
    path_probability(g)

    ##########################################################################
    print('2.4: Done!\n')


def path_probability(g, n=100, sample_set: np.array = None):
    if sample_set is None:
        sample_set = np.array([n.GetId() for n in g.Nodes()])
    count_existence = 0
    for i in range(n):
        scr = dest = 0
        while scr == dest:
            samples = np.random.choice(sample_set, size=2)
            src = samples[0].item()
            dest = samples[1].item()

        count_existence += g.GetShortPath(src, dest, True) > 0
    return count_existence / n


def bfs(graph: snap.PNGraph, starting_node_id, out_direction=True):
    get_neighbors_method = 'GetOutEdges' if out_direction == True else 'GetInEdges'
    search_queue = [starting_node_id]
    discovered = set()
    while search_queue:
        v = search_queue.pop(0)
        v_iter = graph.GetNI(v)
        for neigh_id in getattr(v_iter, get_neighbors_method)():
            if neigh_id not in discovered:
                discovered.add(neigh_id)
                search_queue.append(neigh_id)

    return discovered


if __name__ == "__main__":
    Rnd = snap.TRnd(1234)
    q1_1()
    q1_2()
    q1_3()
    q1_4()
    print("Done with Question 2!\n")
