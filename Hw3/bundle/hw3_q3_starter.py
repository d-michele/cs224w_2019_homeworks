###############################################################################
# CS 224W (Fall 2019) - HW3
# Starter code for Problem 3
###############################################################################
from functools import reduce

import snap
import matplotlib.pyplot as plt
import heapq

# Setup
num_voters = 10000
decision_period = 10


def read_graphs(path1, path2):
    """
    :param - path1: path to edge list file for graph 1
    :param - path2: path to edge list file for graph 2

    return type: snap.PUNGraph, snap.PUNGraph
    return: Graph 1, Graph 2
    """
    ###########################################################################
    # TODO: Your code here!
    Graph1 = snap.LoadEdgeList(snap.PUNGraph, 'graph1.txt', 0, 1)
    Graph2 = snap.LoadEdgeList(snap.PUNGraph, 'graph2.txt', 0, 1)
    ###########################################################################
    return Graph1, Graph2


def initial_voting_state(Graph):
    """
    Function to initialize the voting preferences.

    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: Python dictionary
    return: Dictionary mapping node IDs to initial voter preference
            ('A', 'B', or 'U')

    Note: 'U' denotes undecided voting preference.

    Example: Some random key-value pairs of the dict are
             {0 : 'A', 24 : 'B', 118 : 'U'}.
    """
    voter_prefs = {}
    ###########################################################################
    # TODO: Your code here!
    votes_assignment = {}
    votes_assignment.update({k: 'A' for k in range(4)})
    votes_assignment.update({k: 'B' for k in range(4, 8)})
    votes_assignment.update({k: 'U' for k in range(8, 10)})
    for n_id in range(Graph.GetNodes()):
        voter_prefs[n_id] = votes_assignment[n_id % 10]
    ###########################################################################
    assert (len(voter_prefs) == num_voters)
    return voter_prefs


def iterate_voting(Graph, init_conf):
    """
    Function to perform the 10-day decision process.

    :param - Graph: snap.PUNGraph object representing an undirected graph
    :param - init_conf: Dictionary object containing the initial voting
                        preferences (before any iteration of the decision
                        process)

    return type: Python dictionary
    return: Dictionary containing the voting preferences (mapping node IDs to
            'A','B' or 'U') after the decision process.

    Hint: Use global variables num_voters and decision_period to iterate.
    """
    curr_conf = init_conf.copy()
    curr_alternating_vote = 'A'
    ###########################################################################
    # TODO: Your code here!
    undecided_voters_ids = [k for k, v in init_conf.items() if v == 'U']
    global decision_period
    for i in range(decision_period):
        for node_id in undecided_voters_ids:
            votes = {'A': 0, 'B': 0}
            ni = Graph.GetNI(node_id)
            for neigh_id in ni.GetOutEdges():
                if curr_conf[neigh_id] == 'U':
                    continue
                votes[curr_conf[neigh_id]] += 1
            if votes['A'] != votes['B']:
                curr_conf[node_id] = max(votes, key=votes.get)
            else:
                curr_conf[node_id] = curr_alternating_vote
                curr_alternating_vote = 'B' if curr_alternating_vote == 'A' else 'A'

    ###########################################################################
    return curr_conf


def sim_election(Graph):
    """
    Function to simulate the election process, takes the Graph as input and
    gives the final voting preferences (dictionary) as output.
    """
    init_conf = initial_voting_state(Graph)
    conf = iterate_voting(Graph, init_conf)
    return conf


def winner(conf):
    """
    Function to get the winner of election process.
    :param - conf: Dictionary object mapping node ids to the voting preferences

    return type: char, int
    return: Return candidate ('A','B') followed by the number of votes by which
            the candidate wins.
            If there is a tie, return 'U', 0
    """
    ###########################################################################
    # TODO: Your code here!
    votes_sum = reduce(aggregate_votes, conf.items(), {'A': 0, 'B': 0})

    if votes_sum['A'] > votes_sum['B']:
        return 'A', votes_sum['A'] - votes_sum['B']
    elif votes_sum['B'] > votes_sum['A']:
        return 'B', votes_sum['B'] - votes_sum['A']
    else:
        return 'U', 0
    ###########################################################################


def aggregate_votes(accumulator, vote):
    accumulator[vote[1]] += 1
    return accumulator


def Q1():
    print("\nQ1:")
    Gs = read_graphs('graph1.txt', 'graph2.txt')  # List of graphs

    # Simulate election process for both graphs to get final voting preference
    final_confs = [sim_election(G) for G in Gs]

    # Get the winner of the election, and the difference in votes for both
    # graphs
    res = [winner(conf) for conf in final_confs]

    for i in range(2):
        print(f'In graph {i + 1}, candidate {res[i][0]} wins by {res[i][1]} votes')


def Q2sim(Graph, k):
    """
    Function to simulate the effect of advertising.
    :param - Graph: snap.PUNGraph object representing an undirected graph
             k: amount to be spent on advertising

    return type: int
    return: The number of votes by which A wins (or loses), i.e. (number of
            votes of A - number of votes of B)

    Hint: Feel free to use initial_voting_state and iterate_voting functions.
    """
    ###########################################################################
    # TODO: Your code here!)
    init_conf = initial_voting_state(Graph)
    for i in range(3000, 3000 + int(k / 100)):
        init_conf[i] = 'A'
    conf = iterate_voting(Graph, init_conf)
    votes_sum = reduce(aggregate_votes, conf.items(), {'A': 0, 'B': 0})

    return votes_sum['A'] - votes_sum['B']
    ###########################################################################


def find_min_k(diffs):
    """
    Function to return the minimum amount needed for A to win
    :param - diff: list of (k, diff), where diff is the value by which A wins
                   (or loses) i.e. (A-B), for that k.

    return type: int
    return: The minimum amount needed for A to win
    """
    ###########################################################################
    # TODO: Your code here!
    for k, diff in diffs:
        if diff > 0:
            return k
    return -1
    ###########################################################################


def makePlot(res, title):
    """
    Function to plot the amount spent and the number of votes the candidate
    wins by
    :param - res: The list of 2 sublists for 2 graphs. Each sublist is a list
                  of (k, diff) pair, where k is the amount spent, and diff is
                  the difference in votes (A-B).
             title: The title of the plot
    """
    Ks = [[k for k, diff in sub] for sub in res]
    res = [[diff for k, diff in sub] for sub in res]
    ###########################################################################
    # TODO: Your code here!
    idx = 1
    for x, y in zip(Ks, res):
        # plt.plot(res[0])
        # x, y = zip(*graph_list)
        plt.plot(x, y, label='Graph ' + str(idx))
        idx += 1
    ###########################################################################
    plt.plot(Ks[0], [0.0] * len(Ks[0]), ':', color='black')
    plt.xlabel('Amount spent ($)')
    plt.ylabel('#votes for A - #votes for B')
    plt.title(title)
    plt.legend()
    plt.show()


def Q2():
    print("\nQ2:")
    # List of graphs
    Gs = read_graphs('graph1.txt', 'graph2.txt')

    # List of amount of $ spent
    Ks = [x * 1000 for x in range(1, 10)]

    # List of (List of diff in votes (A-B)) for both graphs
    res = [[(k, Q2sim(G, k)) for k in Ks] for G in Gs]

    # List of minimum amount needed for both graphs
    min_k = [find_min_k(diff) for diff in res]

    formatString = "On graph {}, the minimum amount you can spend to win is {}"
    for i in range(2):
        print(formatString.format(i + 1, min_k[i]))

    makePlot(res, 'TV Advertising')


def Q3sim(Graph, k):
    """
    Function to simulate the effect of a dining event.
    :param - Graph: snap.PUNGraph object representing an undirected graph
             k: amount to be spent on the dining event

    return type: int
    return: The number of votes by which A wins (or loses), i.e. (number of
            votes of A - number of votes of B)

    Hint: Feel free to use initial_voting_state and iterate_voting functions.
    """
    ###########################################################################
    # TODO: Your code here!
    init_conf = initial_voting_state(Graph)
    node_deg_heap = []
    for key, value in init_conf.items():
        ni = Graph.GetNI(key)
        heapq.heappush(node_deg_heap, (ni.GetDeg(), key))
    topk = heapq.nlargest(int(k / 1000), node_deg_heap, key=lambda t: [t[0], -t[1]])
    for _, nid in topk:
        init_conf[nid] = 'A'
    conf = iterate_voting(Graph, init_conf)
    votes_sum = reduce(aggregate_votes, conf.items(), {'A': 0, 'B': 0})

    return votes_sum['A'] - votes_sum['B']
    ###########################################################################


def Q3():
    print("\nQ3:")
    # List of graphs
    Gs = read_graphs('graph1.txt', 'graph2.txt')

    # List of amount of $ spent
    Ks = [x * 1000 for x in range(1, 10)]

    # List of (List of diff in votes (A-B)) for both graphs
    res = [[(k, Q3sim(G, k)) for k in Ks] for G in Gs]

    # List of minimum amount needed for both graphs
    min_k = [find_min_k(diff) for diff in res]

    formatString = "On graph {}, the minimum amount you can spend to win is {}"
    for i in range(2):
        print(formatString.format(i + 1, min_k[i]))

    makePlot(res, 'Wining and Dining')


def Q4():
    """
    Function to plot the distributions of two given graphs on a log-log scale.
    """
    print("\nQ4:")
    ###########################################################################
    # TODO: Your code here!
    Gs = read_graphs('graph1.txt', 'graph2.txt')

    idx = 1
    for g in Gs:
        deg_occ = {}
        for i in range(g.GetNodes()):
            d = g.GetNI(i).GetDeg()
            deg_occ[d] = deg_occ.setdefault(d, 0) + 1
        deg_occ = dict(sorted(deg_occ.items(), key=lambda t: t[0]))
        maxdeg = max(deg_occ.keys())
        xs = list(range(maxdeg + 1))
        ys = [deg_occ[x] if x in deg_occ else 0 for x in xs]
        plt.plot(xs, ys, label='Graph ' + str(idx))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('degree')
        plt.ylabel('frequency')
        plt.title('log')
        idx += 1
    plt.legend()
    plt.show()
    ###########################################################################


def main():
    Q1()
    Q2()
    Q3()
    Q4()


if __name__ == "__main__":
    main()
