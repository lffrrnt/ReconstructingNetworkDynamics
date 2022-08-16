import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import recons_library as rc
import pickle

#eta = float(sys.argv[1])
#realization_number = int(sys.argv[1])
gamma = float(sys.argv[1])

#storage = {"adj_matrix": [], 
#       "laplacian_matrix": [],
#       "delta": [],
#       "degrees": [],
#       "in_degrees": [], 
#       "data": [], 
#       "sync_error": [], 
#       "predicted_series": [], 
#       "Fx": [],
#       "Y_hub": [],
#       "Y": [],
#       "predicted_laplacian": [],
#       "recons_error": [],
#       "old_error": []}

storage = {"adj_matrix": [],
        "laplacian_matrix": [],
        "delta": [],
        "data": [],
        "dist_matrix": [],
        "Fx": [],
        "Y_hub": [],
        "Y": [],
        "predicted_laplacian": []}

#filename="store.pkl"

filename="statistics_data/statistics_gamma_"+str(gamma)+".pkl"
f = open(filename, 'wb')

# integrator parameters
time = 500
transient = 20000
#control parameters
mu=0.001; sigma=0.001; beta=4.1;
#system size
n = 1000 # system size of connected Kasthuri_net
m = 2 #dimension

#add noise on only x-component
x_noise = np.zeros((n*m))
for i in range(n*m):
    if i%2 == 0:
        x_noise[i] = 1
# coupling
C = .1
h = np.eye(m)
h[1,1]=0; #x-coupling

#L = np.loadtxt("Kasthuri_laplacian_connected.txt")
#A = np.loadtxt("Kasthuri_adjajency_connected.txt")
#in_degrees = np.loadtxt("indegree_connected.txt")
#delta = np.max(in_degrees)

for i in range(40):

        G, A, k_in, L, delta = rc.network_generate(n,eta=0.2)
        #G, A, L, degrees, delta = rc.undirected_network(n)

        storage["adj_matrix"].append(A)
        storage["laplacian_matrix"].append(L)
        storage["delta"].append(delta)
#       storage["degrees"].append(degrees)
#       storage["in_degrees"].append(in_degrees)

        x = rc.data_generate(n,m,transient,time,beta,mu,sigma,C,L,delta,h,gamma,x_noise)
        #synchronization = rc.sync_error(x)

        storage["data"].append(x)
        #storage["sync_error"].append(synchronization)

        X,dx = rc.data_split(n,x)

        pred_models = rc.predicted_models(n,X,dx)

        #pred_series = rc.predicted_series(n,X,pred_models)

        #storage["predicted_series"].append(pred_series)

        corr_gt, distance_matrix, s, s_gt, hub_id, ld_id = rc.similarity(n,x,pred_models,k_in)

        storage["dist_matrix"].append(distance_matrix)

        #F_x = rc.predicted_local_dynamics(n, sindy_lib, dx, ld_id, hub_id, true_names, x)

        F_x = rc.local_dynamics_function(n, m, time, x,beta,mu,sigma)
        
        storage["Fx"].append(F_x)

        Y_hub, Y = rc.coupling_effect(dx, hub_id, F_x)

        storage["Y_hub"].append(Y_hub)
        storage["Y"].append(Y)

        L_pred = rc.reconstruction(n,m,time,X,Y)
        
        storage["predicted_laplacian"].append(L_pred)
        
#       error = rc.rmse(L, L_pred, C, delta)

#       storage["recons_error"].append(error)

#       print(error)

pickle.dump(storage,f)
f.close()

