import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Any, Dict, Optional
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from scipy.special import softmax




class TFEst:
             
    def __init__(self, L, var, d):
        self.L = L
        self.var = var
        self.d = d

    def calculate_optimal_mu(self, omega):
        # Step 1: Calculate ωω^T
        omega_outer = np.outer(omega, omega)

        # Step 2: Calculate the exponential term exp(d * ωω^T)
        exp_term = np.exp(self.d * omega_outer)

        # Step 3: Combine to create A
        A = omega_outer + (1 + self.var) * (1 / self.L) * exp_term

        # Step 4: Calculate the inverse of A and multiply by omega
        mu_optimal = np.linalg.inv(A).dot(omega)

        return mu_optimal

    def Est(self, Z_x, Z_y, xq, omega, mu = None):
        
        H = len(omega)
        if mu is None:
            mu = self.calculate_optimal_mu(omega)

        y_hat = 0
        mu_ls = []

        for h in range(H):
            # Step 1: Calculate ω^(h) * Z_x^T * x_q
            inner_product = omega[h] * Z_x.T.dot(xq)

            # Step 2: Apply softmax to inner product
            softmax_result = softmax(inner_product)

            # Step 3: Calculate μ^(h) * Z_y * softmax_result
            y_hat += mu[h] * np.dot(Z_y.T, softmax_result)
            mu_ls.append(mu[h])

        return y_hat#, mu_ls

    def approx_loss(self, omega, mu):
        """
        Calculate the value of L_tilde given vectors omega and mu.

        Parameters:
        - omega: numpy array, the vector ω.
        - mu: numpy array, the vector μ.

        Returns:
        - L: the calculated value of Loss.
        """
        # Step 1: Calculate the first term
        term1 = 1 +self.var - 2 * np.dot(mu.T, omega)

        # Step 2: Calculate ωω^T
        omega_outer = np.outer(omega, omega)

        # Step 3: Calculate the exponential term exp(d * ωω^T)
        exp_term = np.exp(self.d * omega_outer)

        # Step 4: Combine to calculate the inner term
        inner_term = omega_outer + (1 + self.var) * (1 / self.L) * exp_term

        # Step 5: Calculate μ^T * inner_term * μ
        term2 = np.dot(mu.T, np.dot(inner_term, mu))

        # Step 6: Combine all terms
        L = term1 + term2

        return L
    
    def GDEst(self, Z_x, Z_y, xq, omega, mu):
        y_hat = 2 * omega * mu * np.dot(Z_y, Z_x.T.dot(xq))
        return y_hat/self.L#, mu_ls

    def DGDEst(self, Z_x, Z_y, xq, omega, mu):
        #y_hat = 2 * omega * mu * np.dot(Z_y, (Z_x-Z_x.mean(axis=0)).T.dot(xq))
        y_hat = 2 * omega * mu * np.dot(Z_y, Z_x.T.dot(xq)) - 2 * omega * mu * Z_y.sum() * Z_x.mean(axis=1).dot(xq)
        return y_hat/self.L#, mu_ls

    def RidgeEst(self, Z_x, Z_y, xq):
        lambda_I = self.var * self.d * np.eye(Z_x.shape[0])  # d x d identity matrix
        # Ridge regression closed-form solution
        beta = np.linalg.inv(np.dot(Z_x, Z_x.T) + lambda_I).dot(Z_x).dot(Z_y)
        return np.dot(xq, beta)


# write a data generator
# X: d * n matrix  with each column iid from N(0,1) 
# y: n * 1 vector defined by beta^T X + epsilon, where beta is d * 1 vector and epsilon is iid from N(0,var)
def data_generator(n, d, beta, var):
    X = np.random.normal(0, 1, (d, n))
    epsilon = np.random.normal(0, np.sqrt(var), n)
    y = np.dot(beta.T, X) + epsilon
    
    xq = np.random.normal(0, 1, d)
    yq = np.dot(beta.T, xq) + np.random.normal(0, np.sqrt(var))

    return X, y, xq, yq

def ridge_loss(var, xi):
    # Components of the formula
    term1 = var
    term2 = (xi - 1) / xi
    inner_sqrt_term = 4 * var + (var + (1 - xi) / xi) ** 2
    sqrt_term = np.sqrt(inner_sqrt_term)
    
    # Final formula
    result = 0.5 * (term1 + term2 + sqrt_term)
    return result

def generate_softmax_histogram(L, d, scale=1.0):
    # Step 1: Generate L+1 vectors of dimension d from a standard Gaussian distribution
    vectors = np.random.randn(L + 1, d)

    # Step 2: Compute the dot products of each of the first L vectors with the (L+1)th vector
    x_last = vectors[-1]  # The (L+1)th vector
    dot_products = np.array([np.dot(vectors[i], x_last) for i in range(L)])

    # Scale the dot products
    dot_products *= scale

    # Step 3: Calculate the softmax values for these dot products
    exp_values = np.exp(dot_products)
    softmax_values = exp_values / np.sum(exp_values)

    return softmax_values

def plot_norm_histogram(L, d, n, scale=1.0):
    norms = []

    for _ in range(n):
        # Generate softmax vector and calculate its norm
        softmax_vector = generate_softmax_histogram(L, d, scale)
        norm = np.linalg.norm(softmax_vector)
        norms.append(norm)

    return norms

def plot_norm_2histogram(L, d, n, scale=1.0):
    norms = []

    for _ in range(n):
        # Generate softmax vector and calculate its norm
        softmax_vector = generate_softmax_histogram(L, d, scale)
        norm = np.linalg.norm(softmax_vector)
        norms.append(norm**2)

    return norms

# Define the loss function and its gradients
def loss_function(mu, omega, sigma2, L, d):
    omega = np.array(omega).reshape(-1, 1)
    mu = np.array(mu).reshape(-1, 1)

    outer_omega = np.outer(omega, omega.T)
    exp_term = np.exp(d * outer_omega)
    matrix = outer_omega + (1 + sigma2) / L * exp_term

    return (
        1 +sigma2 - 2 * mu.T @ omega + mu.T @ matrix @ mu
    ).item()/2  # Convert 1x1 matrix to scalar

def gradients(mu, omega, sigma2, L, d):
    omega = np.array(omega).reshape(-1, 1)
    mu = np.array(mu).reshape(-1, 1)

    outer_omega = np.outer(omega, omega.T)
    exp_term = np.exp(d * outer_omega)
    matrix = outer_omega + (1 + sigma2) / L * exp_term

    grad_mu = -2 * omega + 2 * matrix @ mu

    grad_omega = (
        -2 * mu + 2 * (mu.T @ omega) * mu +
        (2 * d * (1 + sigma2) / L) * (exp_term * (mu @ mu.T)) @ omega #
    )

    return grad_mu.flatten()/2, grad_omega.flatten()/2

# Gradient flow simulation
def gradient_flow(mu_init, omega_init, lr, steps, sigma2, L, d):
    mu = np.array(mu_init, dtype=float)
    omega = np.array(omega_init, dtype=float)

    mu_history = [mu.copy()]
    omega_history = [omega.copy()]
    loss_history = [loss_function(mu, omega, sigma2, L, d)]

    for _ in range(steps):
        grad_mu, grad_omega = gradients(mu, omega, sigma2, L, d)

        # Update parameters
        mu -= lr * grad_mu
        omega -= lr * grad_omega

        # Store history for plotting
        mu_history.append(mu.copy())
        omega_history.append(omega.copy())
        loss_history.append(loss_function(mu, omega, sigma2, L, d))

    return np.array(mu_history), np.array(omega_history), np.array(loss_history)

def calculate_mu_star(omega, sigma2, L, d):
    """
    Calculate \mu^*(\omega) based on the given formula:
    
    \mu^*(\omega) = (\omega \omega^T + (1 + \sigma^2) \cdot L^{-1} \cdot \exp(d \omega \omega^T))^{-1} \omega

    Parameters:
        omega (array): 1D numpy array representing the \omega vector.
        sigma2 (float): Variance parameter.
        L (float): Scalar parameter.
        d (float): Scalar parameter for the exponential term.

    Returns:
        mu_star (array): 1D numpy array representing \mu^*(\omega).
    """
    omega = np.array(omega).reshape(-1, 1)  # Ensure omega is a column vector

    # Compute \omega \omega^T (outer product)
    outer_omega = np.dot(omega, omega.T)

    # Compute the matrix exponential term
    exp_term = np.exp(d * outer_omega)

    # Compute the full matrix: \omega \omega^T + (1 + \sigma^2) / L \cdot \exp(d \omega \omega^T)
    matrix = outer_omega + (1 + sigma2) / L * exp_term

    # Calculate \mu^*(\omega): matrix_inv \cdot \omega
    mu_star = np.linalg.inv(matrix).dot(omega)
    

    return mu_star.flatten()  # Return as a 1D array


