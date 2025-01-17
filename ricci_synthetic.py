import networkx as nx
import hypernetx as hnx
import networkit as nk

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, 'HGRicciCurvature')
from OllivierRicci_hypergraphs_edges import OllivierRicciHypergraphEdges
from OllivierRicci_hypergraphs_nodes import OllivierRicciHypergraphNodes

from sklearn import metrics
import itertools

def my_surgery(GT_labels, G_origin: nx.Graph(), G_clique:nx.Graph(), weight="weight", cut=0):
    """A simple surgery function that remove the edges with weight above a threshold

    Parameters
    ----------
    G_origin : NetworkX graph
        A graph with ``weight`` as Ricci flow metric to cut.
    weight: str
        The edge weight used as Ricci flow metric. (Default value = "weight")
    cut: float
        Manually assigned cutoff point.

    Returns
    -------
    G : NetworkX graph
        A graph after surgery.
    """
    G = G_origin.copy()
    w = nx.get_edge_attributes(G, weight)

    assert cut >= 0, "Cut value should be greater than 0."
    if not cut:
        cut = (max(w.values()) - 1.0) * 0.6 + 1.0  # Guess a cut point as default

    to_cut = []
    for n1, n2 in G.edges():
        if G[n1][n2][weight] > cut:
            to_cut.append((n1, n2))
            
        
    print("*************** Surgery time ****************")
    print("* Cut %d edges." % len(to_cut))
    G.remove_edges_from(to_cut)
    print("* Number of nodes now: %d" % G.number_of_nodes())
    print("* Number of edges now: %d" % G.number_of_edges())
    cc = list(nx.connected_components(G))
    
    clustering_labels=-1+np.zeros(len(GT_labels))
    for k, component in enumerate(cc):
        clustering_labels[list(component)]=k
    
    
    #Randomly assign remaining nodes
    original_cc=list(nx.connected_components(G_clique))
    N_classes = max(GT_labels)+1
    for component in original_cc[1:]:
        #random_label=0
        random_label=np.random.randint(N_classes)
        clustering_labels[list(component)]=random_label
    print('* Cut value', cut)
    print("* Modularity now: %f " % nx.algorithms.community.quality.modularity(G, cc))
    print("* NMI now: %f " % metrics.normalized_mutual_info_score(clustering_labels, GT_labels))
    print("Number of cluster:" ,len(cc))
    print("*********************************************")

    return G

def pairs_to_hyp(G:nx.Graph(), H:hnx.Hypergraph, weight='weight', agg_type='max'):
    N_edges = len(H.edges)
    hyp_curvs = np.ones(N_edges)
    for k in range(N_edges):
        if len(H.edges[k])!=1:
            hyp_edge = H.edges[k]
            curv_per_hyp=[]
            for d in list(itertools.combinations(hyp_edge, 2)):
                if d[0] in G and d[1] in G:
                    curv_per_hyp.append(G[d[0]][d[1]][weight])
                else:
                    curv_per_hyp.append(0)
            curv_per_hyp=np.array(curv_per_hyp)
            if agg_type=='max':
                hyp_curvs[k]=np.max(curv_per_hyp)
            if agg_type=='avg':
                hyp_curvs[k]=np.average(curv_per_hyp)
            if agg_type=='min':
                hyp_curvs[k]=np.min(curv_per_hyp)
    return hyp_curvs

def surgery_HG(GT_labels, H:hnx.Hypergraph, G_clique_Ricci:nx.Graph(), weight='weight', curv_agg_type = 'max', cut=0): #trash all the weights, only focus on connected components

    N_nodes = len(H.nodes)
    N_edges = len(H.edges)
    hyp_curvs = np.zeros(N_edges)
    for k in range(N_edges):
        if len(H.edges[k])==1:
            hyp_curvs[k]=10 #single hyperedges won't be cut
        else:
            hyp_edge = H.edges[k]
            curv_per_hyp=[]
            for d in list(itertools.combinations(hyp_edge, 2)):
                if d[0] in G_clique_Ricci and d[1] in G_clique_Ricci:
                    curv_per_hyp.append(G_clique_Ricci[d[0]][d[1]][weight])
                else:
                    curv_per_hyp.append(0)
            curv_per_hyp=np.array(curv_per_hyp)
            if curv_agg_type=='max':
                hyp_curvs[k]=np.max(curv_per_hyp)
            if curv_agg_type=='avg':
                hyp_curvs[k]=np.average(curv_per_hyp)
            if curv_agg_type=='min':
                hyp_curvs[k]=np.min(curv_per_hyp)
    edges_to_keep = list(np.argwhere(hyp_curvs<cut).T[0])

    new_edges =[]
    for k in edges_to_keep:
        new_edges.append(H_edges[k])
    for node in range(N_nodes):
        new_edges.append([node])
    
    new_HG = hnx.Hypergraph(new_edges)
    new_HG_adj_clique = new_HG.adjacency_matrix().todense()
    new_G=nx.Graph(new_HG_adj_clique)
    
    cc = list(nx.connected_components(new_G))
    
    clustering_labels=-1+np.zeros(len(GT_labels))
    for k, component in enumerate(cc):
        clustering_labels[list(component)]=k

    print('* Cut value', cut)
    print("* Modularity now: %f " % nx.algorithms.community.quality.modularity(new_G, cc))
    print("* NMI now: %f " % metrics.normalized_mutual_info_score(clustering_labels, GT_labels))
    print("Number of cluster:" ,len(cc))
    print("*********************************************")
    
    return hyp_curvs

#%% Gen SBM

N_nodes=100
p_inter = 0.0
p_intra = 0.3
N_blocks = 2
N_corrupted_edges = 30
size_corrupted = 10

ratios=[[p_intra, p_inter], [p_inter, p_intra]]
Y=np.zeros(N_nodes)
Y[:N_nodes//2]=1

G_SBM = nx.stochastic_block_model([N_nodes//N_blocks]*N_blocks, ratios)
SBM_edges = np.sort(list(G_SBM.edges))
H_edges = []
for e in SBM_edges: #Add true edges from the SBM
    H_edges.append(e)

for k in range(N_corrupted_edges):
    N1 = np.random.choice(N_nodes//N_blocks, size=size_corrupted//N_blocks, replace=False)
    N2 = N_nodes//N_blocks+np.random.choice(N_nodes//N_blocks, size = size_corrupted-size_corrupted//N_blocks, replace=False)
    e = np.concatenate((N1, N2))
    H_edges.append(e)
    
H=hnx.Hypergraph(H_edges)
N_edges = len(H_edges)
h_weights = np.ones(N_edges)

Nodes=np.arange(N_nodes)
A=nx.adjacency_matrix(G_SBM).todense()
H_adj = H.adjacency_matrix().todense()

G_clique=nx.Graph(H_adj)

#%% Compute curvatures

orc_nodes=OllivierRicciHypergraphNodes(H, transport_weight='none', alpha=0.5, base=np.e, exp_power=0, verbose='DEBUG')
orc_nodes.compute_ricci_curvature()
G_orc_nodes=orc_nodes.G.copy()

A_nodes = nx.adjacency_matrix(G_orc_nodes, weight='ricciCurvature').todense()

orc_nodes.compute_ricci_flow(iterations=20)
G_rf_nodes = orc_nodes.G.copy()
A_rf_nodes = nx.adjacency_matrix(G_rf_nodes).todense()

print('Ricci flow on nodes computed')

orc_edges=OllivierRicciHypergraphEdges(H=H, hyperedge_weights=h_weights, transport_weight='none', base=np.e, exp_power=0, verbose='TRACE')
orc_edges.compute_ricci_curvature()
G_orc_edges=orc_edges.G.copy()

A_edges = nx.adjacency_matrix(G_orc_edges, weight='ricciCurvature').todense()
  
print('Curvature computed')

orc_edges.compute_ricci_flow(iterations=20)
G_rf_edges = orc_edges.G.copy()
A_rf_edges = nx.adjacency_matrix(G_rf_edges).todense()

LG_adj = nk.algebraic.adjacencyMatrix(orc_edges.LG).todense()
#%%
for cut in np.arange(0.9, 1.2, 0.02):
    my_surgery(Y, G_origin=G_rf_nodes,G_clique=G_clique, cut=cut)
    
  #%%
for cut in np.arange(0.9, 1.2, 0.01):
    hyp_curvs = surgery_HG(Y, H, G_rf_nodes, curv_agg_type='max', cut=cut)
