# The 1+1-dimensional diffusion equation
# on a rectangular grid of size L x (T*dt)
# L = 1 
# initial conditions  u(x,0) = g(x)
# boundary conditions u(0,t) = u(L,t) = 0

import math
import numpy as np

np.set_printoptions(threshold=np.nan)

def SpatialStep(alpha,u,uPrev,N):
    for x in range(1,N+1): #loop from i=1 to i=N
        u[x] = alpha*uPrev[x-1] + (1.0-2*alpha)*uPrev[x] + alpha*uPrev[x+1]

def ExplicitSolver(alpha,u,N,T):
    # Forward Euler sheme
    for t in range(1,T):
        SpatialStep(alpha,u[t],u[t-1],N)

def g(x):
    return np.sin(math.pi*x)

def main():
    # Number of integration points along x-axis
    N       =   10
    # Spatial step length
    dx = 1/float(N+1)
    print("dx", dx)
    # Step length in time
    dt      =   0.001
    print("dt", dt)
    print("convergence", (dt/(dx**2))) 
    # Number of time steps till final time 
    T       =   100
    if (dt/(dx**2) > 0.5):
        print("Convergence does not work!!!")
        exit(0)
    u = np.zeros((T,N+2),np.double)
    print(np.shape(u))
    #(x,dx) = np.linspace (0,1,N+2, retstep=True)
    (x,dx) = np.linspace (0,1,N+2, retstep=True)
    print("linspace dx", dx)
    print("X", x)
    alpha = dt/(dx**2)
    #Initial codition
    u[0,:] = g(x)
    u[0,0] = u[0,N+1] = 0.0 #Implement boundaries rigidly
    ExplicitSolver(alpha,u,N,T)
    print(u)

if __name__ == '__main__':
    main()

