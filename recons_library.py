import numpy as np
import networkx as nx
import pysindy as ps
import scipy as sp
from scipy import sparse
from sklearn.linear_model import Lasso
from sklearn.metrics import pairwise_distances

def network_generate(n,eta):
        G = nx.scale_free_graph(n, alpha=0.2, beta=0.3, gamma=0.5) #obtain a directed graph
        #remove self loops
        G.remove_edges_from(nx.selfloop_edges(G))
        #remove repeated links
        removed_list = []
        for edge in G.edges:
                if edge[2] != 0:
                        removed_list.append(edge)
        G.remove_edges_from(removed_list)
        for edge in range(G.number_of_edges()):
                list(G.edges(data=True))[edge][2]["weight"] = np.random.uniform(1.-eta,1.+eta)
        A = nx.adj_matrix(G).todense().T
        k_in = np.zeros(G.number_of_nodes())
        for node in range(G.number_of_nodes()):
                if G.in_degree(node) != 0:
                        k_in[node] = sum(list(G.in_edges(node, data=True))[i][2]["weight"] for i in range(G.in_degree(node)))
        #weighted laplacian matrix
        L = np.diag(k_in) - A
        delta = np.max(k_in)
        #degrees = np.array(G.degree)[:,1] #in+out degree
        #in_degrees = np.array(G.in_degree)[:,1] #in degree
        #out_degrees = np.array(G.out_degree)[:,1] #out degree
        return G, A, k_in, L, delta

def undirected_network(n):
    G = nx.barabasi_albert_graph(n,2)
    A = nx.adjacency_matrix(G).todense()
    L = nx.laplacian_matrix(G).todense()
    degrees = np.array(G.degree)[:,1]
    delta = np.max(degrees)
    return G, A, L, degrees, delta

def sync_error(x):
        err = map(lambda t: np.mean(pdist(x[:,:,t])), np.arange(x.shape[2]))
        return np.array(list(err))

def data_generate(n,m,transient,time,beta,mu,sigma,C,L,delta,h,gamma,x_noise):
        def rulkov_map(x):
                x = x.reshape(n,m).T
                return np.asarray([
                        beta/(1+x[0]**2)+x[1],
                        x[1]-mu*x[0]-sigma
                        ]).T.flatten()

        def net_dynamics(x):
                return rulkov_map(x) - (C/delta)*sparse.kron(L,h).dot(x) + x_noise*gamma*np.random.uniform(-1,1,n*m)

        x=np.zeros([2,n,time])
        x[0,:,0] = np.random.uniform(0.0,1.0,n)
        x[1,:,0] = np.random.uniform(0.0,1.0,n)
        x0 = np.vstack((x[0,:,0],x[1,:,0])).flatten()
        #transient for the isolated dynamics
        for l in range(transient):
                x0 = rulkov_map(x0)
        # transient for the coupled dynamics
        for l in range(transient):
                x0 = net_dynamics(x0)
        x=np.zeros([m*n,time])
        x[:,0] = x0
        for l in range(time-1):
                x[:,l+1] = net_dynamics(x[:,l])
        return x.reshape(n,m,time)

def data_split(n,x):
        dx = [] 
        X = []
        for i in range(n):
                dx.append(x[i,:,1:])
                X.append(x[i,:,:-1])
        return np.array(X), np.array(dx)

def predicted_models(n,X,dx):
        library_functions = [
            lambda x : 1./(1+x**2)
        ]
        library_function_names = [
            lambda x : '1/1+' + x + '^2'
        ]
        lib_x = ps.CustomLibrary(library_functions=library_functions, function_names=library_function_names)
        lib_xy = ps.PolynomialLibrary(degree=1, include_bias=True, include_interaction=False)
        lib = ps.GeneralizedLibrary([lib_x, lib_xy], inputs_per_library=np.array([[0,0],[0,1]]))

        coeff = []
        for i in range(n):
            model = ps.SINDy(feature_library=lib, optimizer=ps.STLSQ(threshold=0.0001), discrete_time=True)
            model.fit(X[i].T, x_dot=dx[i].T)
            coeff.append(model.coefficients())
        return np.array(coeff)

#def predicted_equations(x,parameters):
#        return parameters[0][0]/(1+x[0]**2) + parameters[0][1] + parameters[0][2]*x[0] + parameters[0][3]*x[1], 
#        parameters[1][3]*x[1] + parameters[1][2]*x[0] + parameters[1][1]

#def predicted_series(n,x,coeffs):
#        predicted_series = np.zeros_like((x))
#        for i in range(n):
#                predicted_series[i,:,0] = x[0,:,0]
#                for t in range(x.shape[2]-1):
#                        predicted_series[i,:,t+1] = predicted_equations(x[0,:,t],parameters=coeffs[i])
#        return predicted_series

def similarity(n,x,coeff,k_in):
        #Check the correlation between gorund-truth time series u-component
        corr_matrix_gt = np.corrcoef(x[:,0,:], x[:,0,:])[0:n,0:n]
        distance_matrix = pairwise_distances(coeff[:,0,:], metric='seuclidean')
        #similarity analysis
        s = np.sum(distance_matrix, axis=1)
        s_gt = np.sum(np.abs(corr_matrix_gt), axis=1)
        #predicted low-degree node
        print('one of the low degree nodes:', np.argmin(s))
        #predicted hub
        print('predicted hub:', np.argmax(s))
        #check
        print('Predicted low degree node', np.argmin(s), 'has', k_in[np.argmin(s)], 'in connection(s).')
        print('The real hub is:', np.argmax(k_in))
        hub_id = np.argmax(s)
        ld_id = np.argmin(s)
        return corr_matrix_gt, distance_matrix, s, s_gt, hub_id, ld_id

def local_dynamics_function(n,m,time,x,beta,mu,sigma):

        def rulkov_map(x):
                x = x.reshape(n,m).T
                return np.asarray([beta/(1+x[0]**2)+x[1],x[1]-mu*x[0]-sigma]).T.flatten()

        y = x.reshape(n*m,time)
        F_x = np.zeros_like(y)
        for i in range(time):
                F_x[:,i] = rulkov_map(y[:,i])
        Fx = F_x.reshape(n,m,time)
        return Fx

def coupling_effect(dx,hub_id,Fx):
        Y_hub = dx[hub_id,:,:] - Fx[hub_id,:,:-1]
        Y = dx[:,:,:] - Fx[:,:,:-1]
        return Y_hub, Y


def reconstruction(n,m,time,x,y):
        L_predicted = []
        #model = ps.SINDy(feature_library=ps.IdentityLibrary(), optimizer=ps.STLSQ(threshold=0.0001))
        model = ps.SINDy(feature_library=ps.IdentityLibrary(), optimizer=Lasso(alpha=0.001, fit_intercept=False))
        model.fit(x.reshape(n*m,time-1).T, x_dot=y.reshape(n*m,time-1).T, quiet=True)
        L_predicted.append(model.coefficients())
        L_predicted = np.array(L_predicted)[0,:,:]
        #remove kronecker product
        xxx = np.arange(0,n*m,m)
        xx, yy = np.meshgrid(xxx, xxx)
        L_predicted = L_predicted[xx,yy]
        return L_predicted

