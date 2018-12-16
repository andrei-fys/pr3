import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm, trange
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d



def main():
#    # boundary conditions
#    x0 = 0.0 
#    L = 1.0
#    # initial conditions
#    t0 = 0.0
#    # final time
#    t1 = 0.2
#    # spatial dimention discretization
#    Nx = 12
#    # time discretization
#    Nt = 200
# 
#    G_dnn, XX, TT, diffNNanalytic, G_analytic = NNSolver(x0,L,Nx,t0,t1,Nt)
#    # task_d()
####################################################
#    # Number of integration points along x-axis
#    #N       =   10
#    # Spatial step length
#    dx = 1/float(Nx-1)
#    #print("dx", dx)
#    # Step length in time
#    #dt      =   0.001
#    dt      =  t1/float(Nt)
#    print("dt", dt)
#    print("convergence", (dt/(dx**2)))
#    if (dt/(dx**2) > 0.5):
#        print("Convergence does not work!!!")
#        exit(0)
#    u = np.zeros((Nt,Nx),np.double)
#    #print(np.shape(u))
#    (x,dx) = np.linspace (0,1,Nx, retstep=True)
#    #print("linspace dx", dx)
#    #print("X", x)
#    alpha = dt/(dx**2)
#    #Initial codition
#    u[0,:] = g(x)
#    u[0,0] = u[0,Nx-1] = 0.0 #Implement boundaries rigidly
#    ExplicitSolver(alpha,u,Nx-2,Nt)
#    #print(u)
#    np.save('euler.npy', u)
#
#
#    diffNNexplicit = np.abs(u - G_dnn)
#    diffExplicitAnalytic = np.abs(u - G_analytic) 
#
#    MakePlots(XX, TT, G_dnn, G_analytic, u, diffNNanalytic, diffNNexplicit, diffExplicitAnalytic)

    XX = np.load('XX.npy')
    TT = np.load('TT.npy')
    G_analitytic = np.load('analytic.npy')
    diffNNanalytic = np.load('diffNNanalytic.npy')
    diffNNexplicit = np.load('diffNNexplicit.npy')
    diffExplicitAnalytic = np.load('diffExplicitAnalytic.npy')
    G_explicit = np.load('euler.npy')

    # Compare expl solution with analytical solution
    difference = np.abs(G_analitytic - G_explicit)
    print("Max absolute difference: ", np.max(difference))

    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.gca(projection="3d")
    ##ax.set_title("Solution from the deep neural network w/ %d layer" %
    ##             len(num_hidden_neurons))
    #ax.set_title("a")
    #s = ax.plot_surface(XX, TT, G_dnn, linewidth=0,
    #                    antialiased=False, cmap=cm.viridis)
    #ax.set_ylabel("Time $t$")
    #ax.set_xlabel("Position $x$")
    #fig = plt.figure(figsize=(10, 10))
    
    #ax = fig.gca(projection="3d")
    ##ax.set_title("Analytical solution")
    #ax.set_title("b")
    #s = ax.plot_surface(XX, TT, G_analytic, linewidth=0,
    #                    antialiased=False, cmap=cm.viridis)
    #ax.set_ylabel("Time $t$")
    #ax.set_xlabel("Position $x$")
    #fig = plt.figure(figsize=(10, 10))
    
    #ax = fig.gca(projection="3d")
    ##ax.set_title("Explicit")
    #ax.set_title("c")
    #s = ax.plot_surface(XX, TT, G_explicit, linewidth=0,
    #                    antialiased=False, cmap=cm.viridis)
    #ax.set_ylabel("Time $t$")
    #ax.set_xlabel("Position $x$")
    
    #ax = fig.gca(projection="3d")
    #ax.set_title("d")# NN - excact diff
    #s = ax.plot_surface(XX, TT, diffNNanalytic, linewidth=0,
    #                    antialiased=False, cmap=cm.viridis)
    #ax.set_ylabel("Time $t$")
    #ax.set_xlabel("Position $x$")
    
    #ax = fig.gca(projection="3d")
    #ax.set_title("e")# NN - euler diff
    #s = ax.plot_surface(XX, TT, diffNNexplicit, linewidth=0,
    #                    antialiased=False, cmap=cm.viridis)
    #ax.set_ylabel("Time $t$")
    #ax.set_xlabel("Position $x$")
    
    #ax = fig.gca(projection="3d")
    #ax.set_title("f") # Explicit exact diff
    #s = ax.plot_surface(XX, TT, diffExplicitAnalytic, linewidth=0,
    #                    antialiased=False, cmap=cm.viridis)
    #ax.set_ylabel("Time $t$")
    #ax.set_xlabel("Position $x$")
    #plt.show()
 



if __name__ == '__main__':
    main()

