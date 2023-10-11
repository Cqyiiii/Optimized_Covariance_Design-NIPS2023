import networkx as nx
import networkx.algorithms.community as nx_comm
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import optim



# return the clusters this node connects to, which can be used in exposure
def node_connects_cluster(node):
    return set(map(lambda x: inverse_cluster_dict[x], list(g[node]))).union(set([inverse_cluster_dict[node]]))


parser = argparse.ArgumentParser(description='Generate Optimized Covariance Matrix')
parser.add_argument("--resolution", type=int, default=10) # clustering resolution
parser.add_argument("--omega", type=float, default=1)
parser.add_argument("--dataset", type=str, default="stf")
args = parser.parse_args()



if args.dataset == "stf":
    path = 'Dataset/socfb-Stanford3.mtx'
elif args.dataset == "cornell":
    path = 'Dataset/socfb-Cornell5.mtx'

df = pd.read_table(path, skiprows=1, names = ["source", "target"], sep=" ")
g = nx.from_pandas_edgelist(df)

# calculate basic elements
num_nodes = g.number_of_nodes()
num_edges = g.number_of_edges()
degs = [g.degree[i] for i in g.nodes]
avg_deg = sum(degs)/len(degs)


clu_res = args.resolution
clusters = nx_comm.louvain_communities(g, seed = 10, resolution = clu_res)
clusters = sorted(clusters, key = len, reverse=True)
cluster_sizes = list(map(len, clusters))
num_cluster = len(clusters)

# dict: from node to its cluster
inverse_cluster_dict = {
    node: cl for cl in range(num_cluster) for node in clusters[cl]
}

# dict: from node to its connected cluster
node_to_connected_clusters = {
    node: node_connects_cluster(node) for node in range(1, num_nodes + 1)
}

# in-cluster degree sum
influence = np.array([
    sum([
        g.degree[node] for node in cl
    ])
    for cl in clusters
])


for node in g.nodes:
    cl = inverse_cluster_dict[node]
    cnt = sum([1 for neighbour in g[node] if inverse_cluster_dict[neighbour] == cl]) # in the same cluster
    g.nodes[node]['ai'] = cnt
    g.nodes[node]['bi'] =  g.degree[node] - cnt
    

C_edges = np.zeros((num_cluster, num_cluster), dtype=np.int64)

for edge in g.edges:
    ci = inverse_cluster_dict[edge[0]]
    cj = inverse_cluster_dict[edge[1]]
    C_edges[ci,cj] += 1
    C_edges[cj,ci] += 1
    


C_tensor = torch.tensor(C_edges, dtype = torch.float64)
ones_tensor = torch.ones((num_cluster, num_cluster), dtype=torch.float64)
var_coef = torch.matmul(torch.matmul(C_tensor, ones_tensor), C_tensor)

influ_tensor = torch.tensor(influence, dtype=torch.float64)
influ_mat = torch.matmul(influ_tensor.reshape(-1,1), influ_tensor.reshape(1,-1))

L = np.random.randn(num_cluster, num_cluster)
L = torch.tensor(L, requires_grad = True)
U = F.normalize(L, dim=1)

optimizer = optim.Adam([L], lr = 0.1, weight_decay=0)
num_epoch = 200
eps = 1e-6
omega = args.omega 

for epoch in range(num_epoch):
    
    Z = torch.matmul(U, U.T)    
    V = torch.arcsin(Z/(1+eps))/(2*torch.pi)
    bias = (4 * torch.trace(torch.matmul(C_tensor, V)) - torch.sum(C_tensor))/num_nodes    
    var = 8 * (omega**2 + 4) * torch.trace(torch.matmul(influ_mat, (V + 1/4 * ones_tensor )))
    var /= num_nodes**2
    
    mse = bias ** 2 + var

    print("epoch: {}, mse: {:.5f}".format(epoch + 1, mse))
    optimizer.zero_grad()
    mse.backward(retain_graph=True)
    optimizer.step()
    
    U = F.normalize(L, dim=1) # row normalization
    


l = U.detach().numpy()
np.save("optimized_cov/L_matrix_cov_res{}_omega{}_{}.npy".format(clu_res, omega, args.dataset), l)
