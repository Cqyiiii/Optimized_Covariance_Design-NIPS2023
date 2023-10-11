# import torch
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
import numpy.linalg as alg
from scipy.spatial.distance import mahalanobis as mah_dist # u, v, cov_mat

import os
from tqdm import tqdm
from functools import reduce
import argparse




# Utils
## return the clusters this node connects to, which can be used in exposure
def node_connects_cluster(node):
    return set(map(lambda x: inverse_cluster_dict[x], list(g[node]))).union(set([inverse_cluster_dict[node]]))

## calculate between-cluster Mah distance
def cl_index_to_mah(tr_cl_index, ct_cl_index, pre=True):
    if pre:
        tr_covs = np.array([[cl_covariates[i]["in_size"], cl_covariates[i]["in_avg_deg"]] for i in tr_cl_index])
        ct_covs = np.array([[cl_covariates[i]["in_size"], cl_covariates[i]["in_avg_deg"]] for i in ct_cl_index])
        tr_vec = tr_covs.mean(axis=0)
        ct_vec = ct_covs.mean(axis=0)
        return mah_dist(tr_vec, ct_vec, pre_cov_mat)
    else: # after treatment assignment
        calculate_covariates_post_treatment(tr_cl_index)
        tr_covs = np.array([[cl_covariates[i]["in_size"], cl_covariates[i]["in_avg_deg"], cl_covariates[i]["ex_eff_size"], cl_covariates[i]["ex_eff_avg_deg"]] for i in tr_cl_index])
        ct_covs = np.array([[cl_covariates[i]["in_size"], cl_covariates[i]["in_avg_deg"], cl_covariates[i]["ex_eff_size"], cl_covariates[i]["ex_eff_avg_deg"]] for i in ct_cl_index])
        tr_vec = tr_covs.mean(axis=0)
        ct_vec = ct_covs.mean(axis=0)
        
        # pre_cov_mat can be calculated only once, but post_cov_mat depends on treatment assignment, and should be calculated every assignment
        post_covs = np.array([[cl_covariates[i]["in_size"], cl_covariates[i]["in_avg_deg"], cl_covariates[i]["ex_eff_size"], cl_covariates[i]["ex_eff_avg_deg"]] for i in range(num_cluster)])
        post_cov_normalized = post_covs - post_covs.mean(axis = 0)
        post_cov_mat = np.matmul(post_cov_normalized.T, post_cov_normalized)/num_cluster                
        return mah_dist(tr_vec, ct_vec, post_cov_mat)

## covariates that can be calculated after treatment
def calculate_covariates_post_treatment(tr_clusters):
    if not isinstance(tr_clusters, set):
        tr = set(tr_clusters)
    else:
        tr = tr_clusters
    for unit in g.nodes:    
        connected_clusters = node_connects_cluster(unit)
        if connected_clusters.issubset(tr) or connected_clusters.isdisjoint(tr):  # effectively treated/controled
            g.nodes[unit]["eff"] = True
        else:
            g.nodes[unit]["eff"] = False
        
    for i in range(num_cluster):
        cl_covariates[i]["eff_size"] = sum( g.nodes[node]["eff"] for node in clusters[i] )
        cl_covariates[i]["ex_eff_size"] = sum( g.nodes[node]["eff"] for node in clusters[i] if g.nodes[node]["n_cl"] > 1 )  # effectively treated nodes in exterior
        ex_eff_node_deg = [len(g[node]) for node in clusters[i] if g.nodes[node]["n_cl"]>1 and g.nodes[node]["eff"]]
        cl_covariates[i]["ex_eff_avg_deg"] = sum(ex_eff_node_deg) / len(ex_eff_node_deg) if len(ex_eff_node_deg) > 0 else 0


## estimate the generalized propensity score, i.e. the expectation of exposure indicator
def estimate_propensity_score(i, ps_rounds):
    for epoch in range(ps_rounds):
        if randomization_scheme == "ber": # bernoulli
            tr_clusters = np.where(np.random.binomial(1, p, size = len(clusters))>0)[0]
        elif randomization_scheme == "cr": # complete randomization
            tr_clusters = np.random.choice(range(num_cluster), int(num_cluster * p), replace=False, )
        elif randomization_scheme == "dp": # near-optimal dp
            tr_clusters = dp_randomization(block_split)
        elif randomization_scheme == "psr":
            tr_clusters, _ = PSR()
        elif randomization_scheme == "rer":
            tr_clusters = tr_cl_list[np.random.choice(tops, 1)[0]]            
        tr_units = reduce(lambda x, y: x.union(y), [clusters[i] for i in tr_clusters])  
        nx.set_node_attributes(g, 0, "z")
        nx.set_node_attributes(g, {unit:1 for unit in tr_units}, "z")        
        tr = set(tr_clusters)
        for unit in g.nodes:
            connected_clusters = node_to_connected_clusters[unit] 
            if connected_clusters.issubset(tr):
                g.nodes[unit]["eff_tr_cnt_"+str(i)] += 1   




# Potential Outcome Model
## linear potential outcome model
def po_linear_model(graph, alpha=1, beta=1, sigma=0.1, hete=0.5, gamma=2):    
    hete = 0.5
    for i in graph.nodes:
        graph.nodes[i]["y"] = alpha + beta * graph.nodes[i]["z"] + sigma * np.random.normal() + hete * graph.degree[i]/avg_deg + gamma * sum([graph.nodes[ngbr]['z'] for ngbr in graph[i]])/graph.degree[i]  # + np.random.normal(0, 0.2)
        # graph.nodes[i]["y"] = alpha + beta * graph.nodes[i]["z"] + hete * graph.degree[i]/avg_deg + gamma * sum([graph.nodes[ngbr]['z'] for ngbr in graph[i]])
       
## multiplicative potential outcome model        
def po_multiplicative_model(graph, alpha=1, sigma=0.1, delta=1, gamma=2): 
    for i in graph.nodes:
        graph.nodes[i]["y"] = ( (alpha + sigma * np.random.normal()) * graph.degree[i]/avg_deg )  *  (1 + delta * graph.nodes[i]["z"] + gamma * sum([graph.nodes[ngbr]['z'] for ngbr in graph[i]]) / len(graph[i]) )
        # graph.nodes[i]["y"] = ( (alpha + sigma * np.random.normal() ) )  *  (1 + delta * graph.nodes[i]["z"] + gamma * sum([graph.nodes[ngbr]['z'] for ngbr in graph[i]]) / len(graph[i]) )
        
def po_correct_model(graph, alpha=1, beta=1, sigma=0.1, gamma=0.02):
    for i in graph.nodes:
        graph.nodes[i]["y"] = alpha + beta * graph.nodes[i]["z"] + sigma * np.random.normal() + gamma * sum([graph.nodes[ngbr]['z'] for ngbr in graph[i]])
        

# Baseline methods
## dynamic programming for Independent Block Randomization(IBR), generate block split
def near_optimal_dp():
    rev_cl_sz = rev_cluster_sizes.tolist()
    V = np.zeros(num_cluster + 1)
    fr = np.zeros(num_cluster + 1, dtype=np.int32)
    V[1] = rev_cl_sz[0]**2
    for i in range(2, num_cluster + 1):
    # for i in range(2, 3):
        indices = []
        for j in range(1, i+1):
            # calculate gkh.   rev_cluster_sizes[]
            # vk: rev[:k]  vk-h: rev[:(k-h)]
            temp_block = rev_cl_sz[(i-j):i][::-1] # back to decreasing order 
            if j%2 == 0:
                sigma = -1/(j-1)
            else:
                sigma = -1/j                
            cut_p = 0
            while cut_p < j and temp_block[cut_p] > -sigma * sum(temp_block[:cut_p]):
                cut_p+=1                            
            y = temp_block[:cut_p] + [0]*(j-cut_p)
            y = np.array(y)
            corr_mat = sigma * np.ones([j, j]) + (1-sigma) * np.eye(j)            
            gkh = np.dot(y.T, np.dot(corr_mat, y))
            indices.append(gkh + V[i-j])
        indices = np.array(indices)
        min_index = np.argmin(indices)
        fr[i] =  (i-1) - min_index # cutting point
        V[i] = indices[min_index]
    block_split = []
    x = num_cluster
    while x>0:
        block_split.append([fr[x], x])
        x = fr[x]
    block_split = block_split[::-1] # from small to larger cluster    
    return block_split


## randomization of IBR, receive block split and output treated clusters
def dp_randomization(block_split):
    tr_clusters = []
    for block_index in block_split:
        block_size = block_index[1] - block_index[0]            
        if block_size % 2 == 1 :
            block_tr = np.random.choice(range(block_index[0], block_index[1] - 1), size = block_size//2, replace=False).tolist()        
            if np.random.uniform() > 0.5:
                block_tr.append(block_index[1] - 1)
        else:
            block_tr = np.random.choice(range(block_index[0], block_index[1]), size = block_size//2, replace=False).tolist()                
        tr_clusters = tr_clusters + block_tr                    
    tr_clusters = sorted(tr_clusters)    
    return tr_clusters


## randomization of IBR-p, 
def sort_dp_randomization():
    tr_clusters = []
    arg_index_influ = np.argsort(influence)
    if len(arg_index_influ)%2 == 1:
        if np.random.uniform() > 0.5:
            tr_clusters.append(arg_index_influ[0])
        for i in range(num_cluster//2):
            if np.random.uniform() > 0.5:
                tr_clusters.append(2*i+2)
            else:
                tr_clusters.append(2*i+1)
    else:
        for i in range(num_cluster//2):
            if np.random.uniform() > 0.5:
                tr_clusters.append(2*i)
            else:
                tr_clusters.append(2*i+1)
    return tr_clusters
            



## Pairwise Sequential Randomization (PSR)
def PSR(q=0.85):
    tr_clusters = []
    ct_clusters = []
    
    # first 2 clusters assignment
    if np.random.uniform() > 0.5:
        tr_clusters.append(0)
        ct_clusters.append(1)
    else:
        tr_clusters.append(1)
        ct_clusters.append(0)
    K = num_cluster

    for k in range(1, K//2):
        dummy_tr_1 = tr_clusters + [2*k]
        dummy_ct_1 = ct_clusters + [2*k+1]
        dummy_tr_2 = tr_clusters + [2*k+1]
        dummy_ct_2 = ct_clusters + [2*k]
        mah_1 = cl_index_to_mah(dummy_tr_1, dummy_ct_1)
        mah_2 = cl_index_to_mah(dummy_tr_2, dummy_ct_2)

        # with high probability to adopt scheme with low distance    
        random_factor = (np.random.binomial(1, q) == 1)
        if (mah_1 > mah_2 and random_factor) or (mah_1 <= mah_2 and not random_factor):
            tr_clusters.append(2*k+1)
            ct_clusters.append(2*k)
        else:
            tr_clusters.append(2*k)
            ct_clusters.append(2*k+1)
                                    
    if K%2 == 1:    
        if np.random.uniform() > 0.5:
            tr_clusters.append(K-1)        
        else:
            tr_clusters.append(K-1)
            
    return tr_clusters, ct_clusters







## randomization of Optimized Covariance Design (OCD)
def cov_optimal_randomization():
    l = np.load("optimized_cov/L_matrix_cov_res{}_omega{}_{}.npy".format(args.resolution, args.omega, args.dataset))
    Z = (np.sign(np.matmul(l, np.random.randn(num_cluster))) + 1)/2  # mu + Lx, reparameterization
    return np.where(Z==1)[0]
    




# if __name__ == "__main__":


parser = argparse.ArgumentParser(description='Simulation Main')

parser.add_argument("--resolution", type=int, default=10) # clustering resolution
parser.add_argument("--gamma", type=float, default=1.0) 
parser.add_argument("--model", type=str, default="linear") # "linear" or "multi"
parser.add_argument("--omega", type=float, default=1)
parser.add_argument("--dataset", type=str, default="stf") # stf, cornell
args = parser.parse_args()



if args.dataset == "stf":
    path = 'Dataset/socfb-Stanford3.mtx'
elif args.dataset == "cornell":
    path = 'Dataset/socfb-Cornell5.mtx'







# construct graph

df = pd.read_table(path, skiprows=1, names = ["source", "target"], sep=" ")
g = nx.from_pandas_edgelist(df)

# calculate basic elements
num_nodes = g.number_of_nodes()
num_edges = g.number_of_edges()
degs = [g.degree[i] for i in g.nodes]
avg_deg = sum(degs)/len(degs)


# clustering
# generally, we fix the outcome of clustering
clusters = nx_comm.louvain_communities(g, seed = 10, resolution=args.resolution)
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


influence = np.array([
    sum([
        g.degree[node] for node in cl
    ])
    for cl in clusters
])



for i in g.nodes:
    g.nodes[i]["n_cl"] = len(node_to_connected_clusters[i])



# prepare for near-optimal dp
rev_cluster_sizes = cluster_sizes[::-1]
rev_clusters = clusters[::-1]
rev_cluster_sizes = np.array(rev_cluster_sizes)


# prepare for psr
cl_covariates = {
    i : dict() for i in range(num_cluster)
}
# covariates that can be calculated before treatment
for i in range(num_cluster):
    cl_covariates[i]["in_size"] = sum( g.nodes[node]["n_cl"]==1 for node in clusters[i] )
    cl_covariates[i]["ex_size"] = cluster_sizes[i] - cl_covariates[i]["in_size"]  
    in_node_deg = [len(g[node]) for node in clusters[i] if  g.nodes[node]["n_cl"]==1]
    cl_covariates[i]["in_avg_deg"] = sum(in_node_deg) / len(in_node_deg) if len(in_node_deg) > 0 else 0




# covariance matrix of pre-treatment covariates
pre_covs = np.array([[cl_covariates[i]["in_size"], cl_covariates[i]["in_avg_deg"]] for i in range(len(clusters))])
pre_cov_normalized = pre_covs - pre_covs.mean(axis = 0)
pre_cov_mat = np.matmul(pre_cov_normalized.T, pre_cov_normalized)/num_cluster


# basic setting
p = 0.5
num_rounds = 1000
po_model = args.model # linear, multi
schemes = ["ber", "cr", "rer", "psr", "dp", "pair", "cov"]
# schemes = ["cov"]
num_threads = 4

B, alpha = 400, 0.1



# calculation that can be performed only once, in IBR and ReAR
if "dp" in schemes:
    block_split = near_optimal_dp()   

# generate B PSR randomization if rerandomization
if "rer" in schemes:
    top_number = int(B*alpha) 
    tr_cl_list = []
    imba_list = []
    for i in tqdm(range(B)):
        tr_clusters, ct_clusters = PSR()
        imba_list.append(cl_index_to_mah(tr_clusters, ct_clusters, pre=False)) 
        tr_cl_list.append(tr_clusters)
    imba_list = np.array(imba_list)
    tops = np.argpartition(imba_list, top_number)[:top_number] # alpha*B = 50

 
    
# calculate true GATE, we fix beta here
true_gate = 1 + args.gamma

# create experiment outcome record path
outcome_save_path = "./experiment_outcome_{}/".format(args.dataset)
if not os.path.exists(outcome_save_path):
    os.mkdir(outcome_save_path)
    

save_path = outcome_save_path + "res{}_ncl{}_gamma{}_{}.txt".format(args.resolution, num_cluster, args.gamma, args.model)

with open(save_path, "a") as f:
    f.write("\nTrue GATE: {:.5f}\n".format(true_gate))



# repetition
for randomization_scheme in schemes:
    print("Begin scheme {}".format(randomization_scheme))
    estimators = [[], [], []]

    for epoch in tqdm(range(num_rounds)): # every overall treatment assignment                 
        if randomization_scheme == "ber": # bernoulli
            tr_clusters = np.where(np.random.binomial(1, p, size = len(clusters))>0)[0]
        elif randomization_scheme == "cr": # complete randomization
            tr_clusters = np.random.choice(range(num_cluster), int(num_cluster * p), replace=False, )
        elif randomization_scheme == "dp": 
            tr_clusters = dp_randomization(block_split)            
        elif randomization_scheme == "pair": 
            tr_clusters = sort_dp_randomization()        
        elif randomization_scheme == "psr":
            tr_clusters, _ = PSR()
        elif randomization_scheme == "rer":
            tr_clusters = tr_cl_list[np.random.choice(tops, 1)[0]]
        elif randomization_scheme == "cov":
            tr_clusters = cov_optimal_randomization()
            
        # exclude all treated/controlled situation
        if len(tr_clusters) == 0 or len(tr_clusters) == num_cluster:
            continue
        
        # treated units
        tr_units = reduce(lambda x, y: x.union(y), [clusters[i] for i in tr_clusters])  

        # set treatment level
        nx.set_node_attributes(g, 0, "z")
        nx.set_node_attributes(g, {unit:1 for unit in tr_units}, "z")

        if args.model == "linear":
            po_linear_model(g, gamma = args.gamma)
        elif args.model == "multi":
            po_multiplicative_model(g, gamma = args.gamma)
        elif args.model == "correct":
            po_correct_model(g)
        else:
            print("model specification error!")
            break
            
                        
        # estimators 
        ## DIM
        mo1, mo0 = 0, 0
        for unit in g.nodes:
            if g.nodes[unit]['z'] == 1:
                mo1 += g.nodes[unit]['y']
            else:
                mo0 += g.nodes[unit]['y']
                
        DIM_1 = mo1/len(tr_units) - mo0/(len(g) - len(tr_units))
        DIM_2 = (mo1 * 1/(len(tr_clusters)) - mo0 * (1/(num_cluster - len(tr_clusters)))) * (num_cluster / num_nodes)
        HT = 2*(mo1 - mo0)/len(g)
          
        estimators[0].append(DIM_1)
        estimators[1].append(DIM_2)
        estimators[2].append(HT)

    
    with open(save_path, "a") as f:
        f.write(randomization_scheme + "\n")

        for i in range(len(estimators)):
            estimators[i] = np.array(estimators[i])
            mse = ((estimators[i] - true_gate)**2).mean() 
            f.write("Mean: {:.5f}, Bias: {:.5f}, Std: {:.5f}, MSE: {:.5f}\n".format(estimators[i].mean(), estimators[i].mean() - true_gate, estimators[i].std(), mse))
        
        f.write("\n")

    


