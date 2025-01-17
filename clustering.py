import networkx as nx
import hypernetx as hnx

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    print("* ARI now: %f " % metrics.adjusted_rand_score(clustering_labels, GT_labels))
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
                if d[0] in G_rf_nodes and d[1] in G_rf_nodes:
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
        new_edges.append(edges[k])
    for node in range(N_nodes):
        new_edges.append([node])
    
    new_HG = hnx.Hypergraph(new_edges)
    new_HG_adj_clique = new_HG.adjacency_matrix().todense()
    new_G=nx.Graph(new_HG_adj_clique)
    
    cc = list(nx.connected_components(new_G))
    
    clustering_labels=-1+np.zeros(len(GT_labels))
    for k, component in enumerate(cc):
        clustering_labels[list(component)]=k

    NMI = metrics.normalized_mutual_info_score(clustering_labels, GT_labels)
    print('* Cut value', cut)
    print("* Modularity now: %f " % hnx.modularity(H, cc))
    print("* NMI now: %f " % NMI)
    print("* ARI now: %f " % metrics.adjusted_rand_score(clustering_labels, GT_labels))
    print("Number of cluster:" ,len(cc))
    print("*********************************************")
    
    return NMI


def best_NMI(cuts_arr, GT_labels, H:hnx.Hypergraph, G_clique_Ricci:nx.Graph(), weight='weight', curv_agg_type = 'max'):
    NMIs = np.zeros(len(cuts_arr))    
    for k, cut in enumerate(cuts_arr):
        NMIs[k]=surgery_HG(GT_labels, H, G_clique_Ricci, weight, curv_agg_type, cut)
    return np.max(NMIs)

def pairs_to_hyp(G:nx.Graph(), H:hnx.Hypergraph, weight='weight', agg_type='max'):
    N_edges = len(H.edges)
    hyp_curvs = np.ones(N_edges)
    for k in range(N_edges):
        if len(H.edges[k])!=1:
            hyp_edge = H.edges[k]
            curv_per_hyp=[]
            for d in list(itertools.combinations(hyp_edge, 2)):
                if d[0] in G_rf_nodes and d[1] in G_rf_nodes:
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
#Must postprocess data

#%% Load data
dataset='zoo'

if dataset =='mushroom':
    HG = pd.read_pickle(r'dataset/Mushroom/hypergraph.pickle')
    Y = pd.read_pickle(r'dataset/Mushroom/labels.pickle')
if dataset =='pubmed':
    HG = pd.read_pickle(r'dataset/cocitation/pubmed/hypergraph.pickle')
    Y = pd.read_pickle(r'dataset/cocitation/pubmed/labels.pickle')
if dataset =='citeseer':
    HG = pd.read_pickle(r'dataset/cocitation/citeseer/hypergraph.pickle')
    Y = pd.read_pickle(r'dataset/cocitation/citeseer/labels.pickle')
if dataset =='cora_c':
    HG = pd.read_pickle(r'dataset/cocitation/cora/hypergraph.pickle')
    Y = pd.read_pickle(r'dataset/cocitation/cora/labels.pickle')
if dataset =='cora_a':
    HG = pd.read_pickle(r'dataset/coauthorship/cora/hypergraph.pickle')
    Y = pd.read_pickle(r'dataset/coauthorship/cora/labels.pickle')
if dataset =='DBLP':
    HG = pd.read_pickle(r'dataset/coauthorship/dblp/hypergraph.pickle')
    Y = pd.read_pickle(r'dataset/coauthorship/dblp/labels.pickle')
if dataset =='NTU2012':
    HG= pd.read_pickle(r'dataset/NTU2012/hypergraph.pickle')
    Y=pd.read_pickle(r'dataset/NTU2012/labels.pickle')  
if dataset == 'zoo':
    HG= pd.read_pickle(r'dataset/zoo/hypergraph.pickle')
    Y=pd.read_pickle(r'dataset/zoo/labels.pickle')
if dataset == '20news':
    HG= pd.read_pickle(r'dataset/20newsW100/hypergraph.pickle')
    Y=pd.read_pickle(r'dataset/20newsW100/labels.pickle')
if dataset =='football':
    D=np.loadtxt('football/listsMergedHypergraph.mtx').astype(np.int64)[:,:2]
    Y=np.loadtxt('football/classLabels')
    edges=[]
    N_edges = 3601
    for e in range(1, N_edges+1):
        indices = (D[:,1]==e).nonzero()[0]
        if len(indices)<100:
            edges.append(list(D[indices][:,0]-1))
if dataset == 'primary_school':
    with open('contact-primary-school/hyperedges-contact-primary-school.txt', 'r') as file:
        edges = [list(map(int, line.strip().split(','))) for line in file]
    Y=np.loadtxt('contact-primary-school/node-labels-contact-primary-school.txt')
        
    

Y=np.array(Y)
if dataset !='football' and dataset !='primary_school':
    edges = list(HG.values())

# new_edges = []
# for e in edges:
#     if len(e)!=2:
#         new_edges.append(e)
# edges = new_edges
N_nodes = len(Y)
h_weights = np.ones(len(edges))

H=hnx.Hypergraph(edges)

#%%Graph not connected. Preprocess HG. Keep only biggest CC. Inutile ??? Chelou
H_adj_clique = H.adjacency_matrix().todense()
G_clique=nx.Graph(H_adj_clique)
cc = list(nx.connected_components(G_clique))
main_cc=cc[0]

new_edges = []
for e in edges:
    if e[0] in main_cc:
        new_edges.append(e)

new_HG = hnx.Hypergraph(new_edges)

#H=new_HG


#%% Compute curvatures

orc_nodes=OllivierRicciHypergraphNodes(H, transport_weight='none', alpha=0.5, base=np.e, exp_power=0, verbose='DEBUG')
orc_nodes.compute_ricci_curvature()
G_orc_nodes=orc_nodes.G.copy()

A_nodes = nx.adjacency_matrix(G_orc_nodes, weight='ricciCurvature').todense()

orc_nodes.compute_ricci_flow(iterations=10)
G_rf_nodes = orc_nodes.G.copy()
A_rf_nodes = nx.adjacency_matrix(G_rf_nodes).todense()

print('Ricci flow on nodes computed')

orc_edges=OllivierRicciHypergraphEdges(H=H, hyperedge_weights=h_weights, transport_weight='jaccard', base=np.e, exp_power=0, verbose='TRACE')
orc_edges.compute_ricci_curvature()
G_orc_edges=orc_edges.G.copy()

A_edges = nx.adjacency_matrix(G_orc_edges, weight='ricciCurvature').todense()
  
print('Curvature computed')

orc_edges.compute_ricci_flow(iterations=10)
G_rf_edges = orc_edges.G.copy()
A_rf_edges = nx.adjacency_matrix(G_rf_edges).todense()


# global_iterations = 20 #
# for it in range(global_iterations):
#     print(it)
#     orc_edges_alt=OllivierRicciHypergraphEdges(H=H, hyperedge_weights=h_weights, transport_weight='jaccard', base=np.e, exp_power=0, verbose='TRACE')
#     orc_edges_alt.compute_ricci_flow(iterations = 2)
#     G_rf_alt = orc_edges.G.copy()
#     orc_nodes_alt = OllivierRicci(G_rf_alt)
#     orc_nodes_alt.compute_ricci_flow(iterations = 2)
#     G_rf_alt = orc_nodes_alt.G.copy()
#     h_weights = pairs_to_hyp(G_rf_alt, H, weight='weight')

#%%
# for cut in np.arange(0.5, 2, 0.05):
#     my_surgery(Y, G_origin=G_rf_nodes,G_clique=G_clique, cut=cut)
    
 #%%

cuts_arr = np.arange(1, 1.2, 0.01)    
print("Best NMI nodes", best_NMI(cuts_arr, Y, H, G_rf_nodes))
print("Best NMI edges", best_NMI(cuts_arr, Y, H, G_rf_edges))
    
# import pickle
# pickle.dump(G_rf_nodes, open('G_rf_nodes.pickle', 'wb'))
# pickle.dump(G_rf_edges, open('G_rf_edges.pickle', 'wb'))
#%%

# hyp_weights_nodes = pairs_to_hyp(G_orc_nodes, H, weight='ricciCurvature')
# hyp_weights_nodes_rf = pairs_to_hyp(G_rf_nodes, H, weight='weight')
# hyp_weights_edges = pairs_to_hyp(G_orc_edges, H, weight='ricciCurvature')
# hyp_weights_edges_rf = pairs_to_hyp(G_rf_edges, H, weight='weight')

 #%%Save hypedges curvatures to pickle format
# import pickle
# with open('hyp_weights_rc_nodes.pickle', 'wb') as file:
#     pickle.dump(hyp_weights_nodes, file)
# with open('hyp_weights_rf_nodes.pickle', 'wb') as file:
#     pickle.dump(hyp_weights_nodes_rf, file)
# with open('hyp_weights_rc_edges.pickle', 'wb') as file:
#     pickle.dump(hyp_weights_edges, file)
# with open('hyp_weights_rf_edges.pickle', 'wb') as file:
#     pickle.dump(hyp_weights_edges_rf, file)