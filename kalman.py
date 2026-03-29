import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

class KalmanFilter(object):
    def __init__(self, F, Q, H, R, u):
        """
        Initialize the dynamical system models.

        Parameters
        ----------
        F : ndarray of shape (n,n)
            The state transition model.
        Q : ndarray of shape (n,n)
            The covariance matrix for the state noise.
        H : ndarray of shape (m,n)
            The observation model.
        R : ndarray of shape (m,m)
            The covariance matrix for observation noise.
        u : ndarray of shape (n,)
            The control vector.
        """
        # initialize the kf
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.u = u
   

    def estimate(self, x, P, z):
        """
        Compute the state estimates using the Kalman filter.
        If x and P correspond to time step k, then z is a sequence of
        observations starting at time step k+1.

        Parameters
        ----------
        x : ndarray of shape (n,)
            The initial state estimate.
        P : ndarray of shape (n,n)
            The initial error covariance matrix.
        z : ndarray of shape (m,N)
            Sequence of N observations (each column is an observation).

        Returns
        -------
        out : ndarray of shape (n,N)
            Sequence of state estimates (each column is an estimate).
        """
        # Estimate using the update formulas
        out = []
        for z_ in z.T:
            x_k_k_1 = self.F @ x + self.u  
            P_k_k_1 = self.F @ P @ self.F.T + self.Q
            K_k = P_k_k_1 @ self.H.T @ np.linalg.inv(self.H @ P_k_k_1 @ self.H.T + self.R)
            x = x_k_k_1 + K_k @ (z_ - self.H @ x_k_k_1)
            K_k_H = K_k @ self.H
            P = (np.eye(K_k_H.shape[0]) - K_k_H) @ P_k_k_1 
            out.append(x)
        return np.array(out).T

    
    def predict(self, x, k):
        """
        Predict the next k states in the absence of observations.

        Parameters
        ----------
        x : ndarray of shape (n,)
            The current state estimate.
        k : integer
            The number of states to predict.

        Returns
        -------
        out : ndarray of shape (n,k)
            The next k predicted states.
        """
        # Predict using x, F and u
        out = [self.F @ x + self.u]
        for i in range(k-1):
            out.append(self.F @ out[-1] + self.u)
        return np.array(out).T
    

    def rewind(self, x, k):
        """
        Predict the k states preceding the current state estimate x.

        Parameters
        ----------
        x : ndarray of shape (n,)
            The current state estimate.
        k : integer
            The number of preceding states to predict.

        Returns
        -------
        out : ndarray of shape (n,k)
            The k preceding predicted states.
        """
        # Rewind using inverse transformation
        out = [np.linalg.inv(self.F) @ (x - self.u)]
        for i in range(k-1):
            out.append(np.linalg.inv(self.F) @ (out[-1] - self.u))
        return np.array(out).T
    
def evolve(x0, N, F, Q, H, R, u):
    """
    Generate the first N states and observations from the dynamical system.

    Parameters
    ----------
    x0 : ndarray of shape (n,)
        The initial state.
    N : integer
        The number of time steps to evolve.
    F : ndarray of shape (n,n)
        The state transition model.
    Q : ndarray of shape (n,n)
        The covariance matrix for the state noise.
    H : ndarray of shape (m,n)
        The observation model.
    R : ndarray of shape (m,m)
        The covariance matrix for observation noise.
    u : ndarray of shape (n,)
        The control vector.

    Returns
    -------
    states : ndarray of shape (n,N)
        States 0 through N-1, given by each column.
    obs : ndarray of shape (m,N)
        Observations 0 through N-1, given by each column.
    """
    states = [x0] # x0
    obs = [H @ x0 + np.random.multivariate_normal(np.zeros(R.shape[0]), R)] # z0
    for i in range(N-1):
        x_k = F @ states[-1] + u + np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
        z_k = H @ x_k + np.random.multivariate_normal(np.zeros(R.shape[0]), R)
        states.append(x_k)
        obs.append(z_k)
    return np.array(states).T, np.array(obs).T    