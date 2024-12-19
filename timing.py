#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:14:02 2024

@author: olympio
"""

import os
os.chdir('/home/olympio/Documents/GitHub/Hypergraphs/Experiments/Ricci_Hypergraphs')
import networkx as nx
import hypernetx as hnx

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.OllivierRicci_hypergraphs_edges import OllivierRicciHypergraphEdges
from GraphRicciCurvature.OllivierRicci_hypergraphs_nodes import OllivierRicciHypergraphNodes
from GraphRicciCurvature.util import cut_graph_by_cutoff
from sklearn import metrics
import itertools
import time

#%%
K_min = 21
K_max = 31

time_edges = np.zeros(K_max-K_min)
time_nodes = np.zeros(K_max-K_min)

N_nodes=1000
N_edges  = 300
h_weights = np.ones(N_edges+N_nodes-1)
K_list = np.arange(K_min, K_max)
N_runs = 5
for run in range(N_runs):
    print(run)
    for K in K_list:
        #print(K)
        H_edges = []
        for l in range(N_nodes-1):
            H_edges.append([l, l+1])
        for e in range(N_edges):
            edge = np.random.choice(N_nodes, size = K, replace = False)
            H_edges.append(edge)
        H=hnx.Hypergraph(H_edges)
        
        t_0_nodes = time.time()
        orc_nodes=OllivierRicciHypergraphNodes(H, transport_weight='none', alpha=0.5, base=np.e, exp_power=0, verbose='DEBUG')
        orc_nodes.compute_ricci_curvature()
        t_1_nodes = time.time()
        
        t_0_edges = time.time()
        orc_edges=OllivierRicciHypergraphEdges(H=H, hyperedge_weights=h_weights, transport_weight='none', base=np.e, exp_power=0, verbose='TRACE')
        orc_edges.compute_ricci_curvature()
        t_1_edges = time.time()
        
        time_nodes[K-K_min] += t_1_nodes-t_0_nodes
        time_edges[K-K_min] += t_1_edges-t_0_edges
        
time_nodes/=N_runs
time_edges/=N_runs
        
plt.plot(time_nodes, label='nodes')
plt.plot(time_edges, label='edges')
plt.legend()

        
        
        
    
