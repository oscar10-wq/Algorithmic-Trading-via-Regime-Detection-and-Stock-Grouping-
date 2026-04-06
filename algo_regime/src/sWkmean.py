'''
# ============================================================
# CONFIDENTIAL — DO NOT COPY OR DISTRIBUTE
# Author: Oscar Peyron
# University College London — COMP0040
# This source code is submitted as benchmark for testing purposes.
# It is not intended for commercial use or distribution.
# Unauthorised reproduction or sharing is strictly prohibited
# ============================================================

Python file containing sliced Wasserstein k-means clustering functions: 

- (1) Original Sliced Wasserstein function from Luan et al. (2025) "Automated regime 
    classification in multidimensional time series data using sliced Wasserstein k-mean clustering" 

- (2) New Sliced Wasserstein function with Monte Carlo (UnifOrtho) based on Bardenet et al (2025) 
    "Repulsive Monte Carlo on the sphere for the sliced Wasserstein distance".
'''

import numpy as np 
import pandas as pd 
import math
import time
import sympy as sp
import random
import importlib
import algo_regime.src.metrics as mt
from numpy.lib.stride_tricks import sliding_window_view
import scipy.stats as stats

class EmpiricalDistribution:
    def __init__(self, l_m):
        """
        l_m: np.array of shape (h1, d) where h1 is the number of points, d is dimension
        """
        self.l_m = np.asarray(l_m)
        self.h1, self.d = self.l_m.shape
        self.weights = np.ones(self.h1) / self.h1

    def project(self, theta):
        """
        Projects the d-dimensional Dirac masses onto a 1D line defined by theta.
        This is the core operation for Sliced Wasserstein.
        """
        # theta is a vector in R^d. Result is an array of N scalar 1D locations.
        projected_points = np.dot(self.l_m, theta)
        return np.sort(projected_points) # Sort for Wasserstein distance computation

class ProjectedDistribution:
    def __init__(self, x_points):
        self.x_points = np.asarray(x_points)
        self.h1 = len(x_points)
        self.weights = np.ones(self.h1) / self.h1
        self.sorted_atoms = np.sort(self.x_points)
        self.mean_atoms = np.mean(self.x_points)
        self.var_atoms = np.var(self.x_points)
    def return_sorted_atoms(self):
        return self.sorted_atoms # Return re-sorted 1D points for Wasserstein distance computation
    
    def return_mean(self):
        return self.mean_atoms # Return the mean of the 1D points
    
    def return_variance(self):
        return self.var_atoms # Return the variance of the 1D points
#    def return_centroid(self):
#        continue

class sliced_wasserstein_distance:

    # OPTIMIZED CODE - VECTORIZATION  
    def __init__(self, projected_distributions_1, projected_distributions_2, p):
        self.dist1 = projected_distributions_1
        self.dist2 = projected_distributions_2
        self.p = p

    def compute_distance_matrix(self):
        # Extract the pre-sorted arrays into 2D matrices of shape (L, h1)
        X1 = np.array([d.return_sorted_atoms() for d in self.dist1])
        X2 = np.array([d.return_sorted_atoms() for d in self.dist2])
        
        # Vectorized calculation across all L projections simultaneously!
        if self.p == 1:
            return np.mean(np.mean(np.abs(X1 - X2), axis=1))
        elif self.p == 2:
            # np.mean( ... , axis=1) computes the mean across the h1 atoms.
            # The outer np.mean() computes the average across the L projections.
            return np.mean(np.sqrt(np.mean((X1 - X2)**2, axis=1)))
        else:
            return np.mean(np.mean(np.abs(X1 - X2)**self.p, axis=1)**(1/self.p))


def sliced_wasserstein_compute_barycenter(projected_distributions, p):
    """    
    INPUT:
    - projected_distributions: list of projected distibutions   
                               Each element is a list of L ProjectedDistribution objects.
    OUTPUT:
    - centroid: list of length L containing the barycenter ProjectedDistribution objects.
    """
    M_k = len(projected_distributions) 
    if M_k == 0:
        return []
        
    L = len(projected_distributions[0]) 
    
    # OPTIMIZATION 3: Extract all data into a single 3D NumPy array of shape (M_k, L, h1)
    X = np.array([[dist.return_sorted_atoms() for dist in dist_list] for dist_list in projected_distributions])
    
    # Compute the mean/median across the M_k distributions (axis=0) in one vectorized shot
    if p == 1:
        centroid_points = np.median(X, axis=0) # Resulting shape is (L, h1)
    elif p == 2:
        centroid_points = np.mean(X, axis=0)
        
    # Wrap the L centroid projection arrays back into ProjectedDistribution objects
    return [ProjectedDistribution(centroid_points[l]) for l in range(L)]

# Lifting transformation function 

def get_lifted_label_index(price_index, h1, h2):
    """
    Map lifted-window labels back to dates.

    Parameters
    ----------
    price_index : pandas.DatetimeIndex
        Index of the original price series S (before np.diff(log(S))).
    h1 : int
        Lifting window length on returns.
    h2 : int
        Stride.

    Returns
    -------
    pandas.DatetimeIndex
        Dates aligned to each lifted window, using the end date of each window.
    """
    price_index = pd.Index(price_index)
    ret_index = price_index[1:]   # because np.diff(log(S)) removes first row
    positions = np.arange(h1 - 1, len(ret_index), h2)
    return ret_index[positions]

def reorder_labels_by_train_return(labels_train, labels_test, train_price_index, train_prices, h1, h2):
    """
    Reorder train/test labels so regime 0 = worst train-return regime,
    regime K-1 = best train-return regime.

    Parameters
    ----------
    labels_train : ndarray
    labels_test : ndarray or None
    train_price_index : DatetimeIndex
    train_prices : ndarray or DataFrame, shape (N_train, d)
    h1, h2 : lifting parameters

    Returns
    -------
    ordered_train : ndarray
    ordered_test : ndarray or None
    rank_map : dict
    """
    if isinstance(train_prices, pd.DataFrame):
        train_prices_arr = train_prices.values
    else:
        train_prices_arr = np.asarray(train_prices)

    train_idx = get_lifted_label_index(train_price_index, h1, h2)

    # equal-weight simple returns across assets
    train_df = pd.DataFrame(train_prices_arr, index=train_price_index)
    basket_ret = (train_df / train_df.shift(1) - 1).mean(axis=1)
    basket_ret = basket_ret.reindex(train_idx)

    label_series = pd.Series(labels_train, index=train_idx)
    mean_ret_by_cluster = label_series.groupby(label_series).apply(
        lambda g: basket_ret.loc[g.index].mean()
    )

    rank_map = {old: new for new, old in enumerate(mean_ret_by_cluster.sort_values().index)}

    ordered_train = np.array([rank_map[x] for x in labels_train], dtype=int)
    ordered_test = None if labels_test is None else np.array([rank_map[x] for x in labels_test], dtype=int)

    return ordered_train, ordered_test, rank_map

def lifting_transformation(r_S, h1, h2):
    '''
    INPUT : 
    - r_S : numpy (N, d) array of N samples in d dimensions
    - h1 : window size for lifting transfomation 
    - h2 : sliding window offset parameter (stride)
    
    OUTPUT:
    - lifted_samples : numpy (M, h1, d) array
    '''
     
    windows = sliding_window_view(r_S, window_shape=h1, axis=0)
    
    windows = windows.transpose(0, 2, 1)
    
    lifted_samples = windows[::h2]
    
    return lifted_samples.copy()


import numpy as np
import math
from typing import Optional, Tuple, List


def generate_unifortho_theta(d: int, L: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate L orthonormal/random projection vectors in R^d using the
    UnifOrtho construction.

    Returns
    -------
    theta : ndarray of shape (d, L)
    """
    rng = np.random.default_rng(random_state)

    k = math.ceil(L / d)
    theta_blocks = []

    for _ in range(k):
        Z = rng.normal(size=(d, d))
        Q, R = np.linalg.qr(Z)
        lambda_i = np.diag(np.sign(np.diag(R)))
        U_i = Q @ lambda_i
        theta_blocks.append(U_i.T)

    theta = np.concatenate(theta_blocks, axis=0).T[:, :L]   # shape (d, L)
    return theta


def unifortho_projection_vectors_opt(
    S,
    K,
    L,
    h1,
    h2,
    theta: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    return_theta: bool = False,
):
    """
    Build projected empirical distributions for all lifted samples.

    Parameters
    ----------
    S : array-like, shape (N, d)
        Price series (not returns).
    theta : ndarray of shape (d, L), optional
        If provided, use these fixed projection vectors.
        If None, generate a fresh theta.
    return_theta : bool
        If True, also return theta.

    Returns
    -------
    projected_emp_dist : list of length M
        Each element is a list of L ProjectedDistribution objects.
    theta : ndarray of shape (d, L), optional
    """
    S = np.asarray(S)
    r_S = np.diff(np.log(S), axis=0)              # shape (N-1, d)

    l_r_S = lifting_transformation(r_S, h1, h2)   # shape (M, h1, d)
    M = l_r_S.shape[0]
    d = r_S.shape[1]

    if theta is None:
        theta = generate_unifortho_theta(d=d, L=L, random_state=random_state)

    # (M, h1, d) @ (d, L) -> (M, h1, L)
    projections = l_r_S @ theta

    # sort along atom axis
    sorted_projections = np.sort(projections, axis=1)        # (M, h1, L)
    sorted_projections = sorted_projections.transpose(0, 2, 1)  # (M, L, h1)

    projected_emp_dist = [
        [ProjectedDistribution(sorted_projections[m, l]) for l in range(L)]
        for m in range(M)
    ]

    if return_theta:
        return projected_emp_dist, theta
    return projected_emp_dist

def sliced_wasserstein_clustering_conv_loop(projected_emp_dist, K,M, L, epsilon):
    '''
    INPUT : 
    - projected_emp_dist : list of length M, each element is a list of L ProjectedDistribution objects representing the projected distribution for each lifted sample
    - K : number of clusters (regimes)
    - L : number of projections
    - epsilon : converge tolerance

    OUTPUT :
    - projected_emp_dist : list of length M, each element is a list of L ProjectedDistribution objects representing the projected distribution for each lifted sample (same as input, but returned for clarity
    - centroids : list of length K, each element is a list of L ProjectedDistribution objects representing the centroid of that cluster
    - labels : array of length M (number of lifted samples), containing the cluster assignment for
    '''  
    # Initialize K random centroids (1D distributions) for k-means clustering. Choose one distrbituion per cluster for initialization.
    centroids = [0]*K
    for k in range(K):   
        centroids[k] = projected_emp_dist[random.randint(0, M-1)]
    labels = np.full(M, -1) 
    old_centroids = None

    max_iterations = 50 # Set a proper max limit
    for iteration in range(max_iterations):
        #print(f"--- Iteration {iteration + 1} ---")
        
        # Keep track of old labels to check for convergence

        if old_centroids is not None:
            # Check for convergence: if centroids haven't changed significantly, we can stop
            centroid_changes = sum([sliced_wasserstein_distance(old_centroids[k], centroids[k], p=2).compute_distance_matrix() for k in range(K)])
            
            if centroid_changes < epsilon:
                #print("Convergence reached based on centroid changes.")
                break
        
        old_labels = labels.copy()
        old_centroids = centroids.copy() 

        # ==========================================
        # STEP 1: EXPECTATION (Assign to closest centroid)
        # ==========================================
        for m in range(M):
            # Calculate distance to all K centroids
            # Note: k-means theoretically uses squared Wasserstein-2 distance (p=2) 
            # to match the arithmetic mean used in the barycenter step!
            distances_to_centroids = [
                sliced_wasserstein_distance(projected_emp_dist[m], centroids[k], p=2).compute_distance_matrix() 
                for k in range(K)
            ]
            # Assign to the closest centroid
            labels[m] = np.argmin(distances_to_centroids)
            
        # STEP 2: MAXIMIZATION (Update Centroids)
        # ==========================================
        for k in range(K):
            # Gather all projected distributions currently assigned to cluster k
            cluster_k_distributions = [projected_emp_dist[m] for m in range(M) if labels[m] == k]
            
            if len(cluster_k_distributions) > 0:
                centroids[k] = sliced_wasserstein_compute_barycenter(cluster_k_distributions, p=2)

            else:
                print(f"Warning: Cluster {k} became empty. Reinitializing.")
                # Standard k-means fallback: pick a new random point if a cluster dies
                centroids[k] = projected_emp_dist[random.randint(0, M-1)]

    #print("Finished clustering after", iteration + 1, "iterations.", "Final centroid changes:", centroid_changes)
    return projected_emp_dist, centroids, labels

import random

def sliced_wasserstein_clustering_conv_loop_opt(projected_emp_dist, K, M, L, epsilon):
    # Initialize K random centroids
    centroids = [0]*K
    for k in range(K):   
        centroids[k] = projected_emp_dist[random.randint(0, M-1)]
    
    labels = np.full(M, -1) 
    old_centroids = None

    # =========================================================================
    # OPTIMIZATION 2: Pre-extract data into a 3D NumPy array for blazing fast K-means
    # =========================================================================
    # Extract shape: (M, L, h1)
    h1 = len(projected_emp_dist[0][0].return_sorted_atoms())
    X = np.empty((M, L, h1))
    for m in range(M):
        for l in range(L):
            X[m, l, :] = projected_emp_dist[m][l].return_sorted_atoms()

    max_iterations = 50 
    for iteration in range(max_iterations):
        
        # --- OLD CONVERGENCE CHECK ---
        # if old_centroids is not None:
        #     centroid_changes = sum([sliced_wasserstein_distance(old_centroids[k], centroids[k], p=2).compute_distance_matrix() for k in range(K)])
        #     if centroid_changes < epsilon:
        #         break

        # =========================================================================
        # OPTIMIZATION 3: Extract current centroids into array shape (K, L, h1)
        # =========================================================================
        C = np.empty((K, L, h1))
        for k in range(K):
            for l in range(L):
                C[k, l, :] = centroids[k][l].return_sorted_atoms()

        if old_centroids is not None:
            # Vectorized Convergence Check
            # Mean Squared Error between old and new centroids
            C_old = np.empty((K, L, h1))
            for k in range(K):
                for l in range(L):
                    C_old[k, l, :] = old_centroids[k][l].return_sorted_atoms()
            
            centroid_changes = np.sum(np.mean(np.sqrt(np.mean((C_old - C)**2, axis=2)), axis=1))
            if centroid_changes < epsilon:
                #print("Convergence reached based on centroid changes.")
                break

        old_labels = labels.copy()
        old_centroids = centroids.copy() 

        # =========================================================================
        # OPTIMIZATION 4: EXPECTATION (Assign to closest centroid)
        # =========================================================================
        # --- OLD CODE ---
        # for m in range(M):
        #     distances_to_centroids = [
        #         sliced_wasserstein_distance(projected_emp_dist[m], centroids[k], p=2).compute_distance_matrix() 
        #         for k in range(K)
        #     ]
        #     labels[m] = np.argmin(distances_to_centroids)
        
        # --- NEW VECTORIZED EXPECTATION ---
        # Calculate distances using Broadcasting: X is (M, 1, L, h1), C is (1, K, L, h1)
        # This single line computes SW-2 distance for ALL points against ALL centroids!
        diff_sq = (X[:, None, :, :] - C[None, :, :, :]) ** 2
        
        # SW_2 distance: mean over h1 (axis 3), sqrt, mean over L (axis 2)
        sw_dist = np.mean(np.sqrt(np.mean(diff_sq, axis=3)), axis=2) # Shape: (M, K)
        
        # Assign labels instantaneously
        labels = np.argmin(sw_dist, axis=1)
            
        # =========================================================================
        # OPTIMIZATION 5: MAXIMIZATION (Update Centroids)
        # =========================================================================
        # --- OLD CODE ---
        # for k in range(K):
        #     cluster_k_distributions = [projected_emp_dist[m] for m in range(M) if labels[m] == k]
        #     if len(cluster_k_distributions) > 0:
        #         centroids[k] = sliced_wasserstein_compute_barycenter(cluster_k_distributions, p=2)
        #     ...
        
        # --- NEW VECTORIZED MAXIMIZATION ---
        for k in range(K):
            mask = (labels == k)
            if np.any(mask):
                # Barycenter for p=2 is simply the mathematical mean across assigned distributions
                # X[mask] has shape (M_k, L, h1). Mean along axis 0 gives (L, h1)
                barycenter_matrix = np.mean(X[mask], axis=0)

                # Repackage into ProjectedDistribution objects for the next iteration
                centroids[k] = [ProjectedDistribution(barycenter_matrix[l]) for l in range(L)]

            else:
                print(f"Warning: Cluster {k} became empty. Reinitializing.")
                centroids[k] = projected_emp_dist[random.randint(0, M-1)]

    return projected_emp_dist, centroids, labels


def assign_labels_to_centroids(projected_emp_dist, centroids):
    """
    Assign each projected empirical distribution to the nearest existing centroid.

    Parameters
    ----------
    projected_emp_dist : list of length M
    centroids : list of length K

    Returns
    -------
    labels : ndarray of shape (M,)
    """
    M = len(projected_emp_dist)
    K = len(centroids)

    if M == 0:
        return np.array([], dtype=int)

    h1 = len(projected_emp_dist[0][0].return_sorted_atoms())
    L = len(projected_emp_dist[0])

    X = np.empty((M, L, h1))
    for m in range(M):
        for l in range(L):
            X[m, l, :] = projected_emp_dist[m][l].return_sorted_atoms()

    C = np.empty((K, L, h1))
    for k in range(K):
        for l in range(L):
            C[k, l, :] = centroids[k][l].return_sorted_atoms()

    diff_sq = (X[:, None, :, :] - C[None, :, :, :]) ** 2
    sw_dist = np.mean(np.sqrt(np.mean(diff_sq, axis=3)), axis=2)   # shape (M, K)

    labels = np.argmin(sw_dist, axis=1)
    return labels

def fit_swkmeans_train(
    S_train,
    K,
    L,
    epsilon,
    h1,
    h2,
    N_S=5,
    metric="CVaR",
    random_state: Optional[int] = 42,
):
    """
    Fit sWkmeans on the training sample only.

    Returns
    -------
    result : dict with keys
        projected_emp_dist_train
        centroids
        labels_train
        theta
    """
    S_train = np.asarray(S_train)

    projected_emp_dist_train, theta = unifortho_projection_vectors_opt(
        S_train,
        K=K,
        L=L,
        h1=h1,
        h2=h2,
        theta=None,
        random_state=random_state,
        return_theta=True,
    )

    M_train = len(projected_emp_dist_train)
    if M_train == 0:
        raise ValueError("Training sample is too short for the chosen h1/h2.")

    best_mccd = -np.inf
    best_centroids = None
    best_labels = None

    # reproducible but still different initialisations across sims
    py_rng = random.Random(random_state)

    for sim in range(N_S):
        random.seed(py_rng.randint(0, 10**9))

        _, centroids, labels = sliced_wasserstein_clustering_conv_loop_opt(
            projected_emp_dist_train, K, M_train, L, epsilon
        )

        mccd = mt.mean_centroid_centroid_distance(centroids, K, p=2)
        if mccd > best_mccd:
            best_mccd = mccd
            best_centroids = centroids
            best_labels = labels.copy()

    best_centroids, best_labels = choose_label(best_centroids, best_labels, metric, K)

    return {
        "projected_emp_dist_train": projected_emp_dist_train,
        "centroids": best_centroids,
        "labels_train": np.asarray(best_labels, dtype=int),
        "theta": theta,
    }

def predict_swkmeans_test(
    S_test_with_warmup,
    centroids,
    theta,
    h1,
    h2,
):
    """
    Predict regime labels for the test sample using fixed training centroids
    and fixed training projection vectors.

    Parameters
    ----------
    S_test_with_warmup : array-like, shape (N_test_ext, d)
        Test-period prices INCLUDING enough pre-test warmup history.
    centroids : fitted training centroids
    theta : training projection vectors
    h1, h2 : lifting parameters

    Returns
    -------
    result : dict with keys
        projected_emp_dist_test
        labels_test
    """
    S_test_with_warmup = np.asarray(S_test_with_warmup)
    L = theta.shape[1]

    projected_emp_dist_test = unifortho_projection_vectors_opt(
        S_test_with_warmup,
        K=len(centroids),
        L=L,
        h1=h1,
        h2=h2,
        theta=theta,
        random_state=None,
        return_theta=False,
    )

    labels_test = assign_labels_to_centroids(projected_emp_dist_test, centroids)

    return {
        "projected_emp_dist_test": projected_emp_dist_test,
        "labels_test": np.asarray(labels_test, dtype=int),
    }


def fit_predict_swkmeans_train_test(
    S_train,
    S_test_with_warmup,
    train_price_index,
    test_price_index_with_warmup,
    split_date,
    K,
    L,
    epsilon,
    h1,
    h2,
    N_S=5,
    metric="CVaR",
    random_state: Optional[int] = 42,
):
    """
    Train on train sample only, predict on test sample only.

    Returns
    -------
    result : dict
        centroids
        theta
        labels_train
        labels_test
        train_index
        test_index
        projected_emp_dist_train
        projected_emp_dist_test
        rank_map
    """
    train_result = fit_swkmeans_train(
        S_train=S_train,
        K=K,
        L=L,
        epsilon=epsilon,
        h1=h1,
        h2=h2,
        N_S=N_S,
        metric=metric,
        random_state=random_state,
    )

    test_result = predict_swkmeans_test(
        S_test_with_warmup=S_test_with_warmup,
        centroids=train_result["centroids"],
        theta=train_result["theta"],
        h1=h1,
        h2=h2,
    )

    train_index = get_lifted_label_index(train_price_index, h1, h2)
    test_index_ext = get_lifted_label_index(test_price_index_with_warmup, h1, h2)

    split_ts = pd.Timestamp(split_date)
    keep_mask = test_index_ext >= split_ts

    labels_test_raw = test_result["labels_test"][keep_mask]
    test_index = test_index_ext[keep_mask]

    ordered_train, ordered_test, rank_map = reorder_labels_by_train_return(
        labels_train=train_result["labels_train"],
        labels_test=labels_test_raw,
        train_price_index=train_price_index,
        train_prices=S_train,
        h1=h1,
        h2=h2,
    )

    return {
        "centroids": train_result["centroids"],
        "theta": train_result["theta"],
        "labels_train": ordered_train,
        "labels_test": ordered_test,
        "train_index": train_index,
        "test_index": test_index,
        "projected_emp_dist_train": train_result["projected_emp_dist_train"],
        "projected_emp_dist_test": test_result["projected_emp_dist_test"],
        "rank_map": rank_map,
    }
# (1)   Original Sliced Wasserstein function from Luan et al. (2025) "Automated regime
#def sliced_wasserstein_clustering(r_S, K, L, epsilon, h1, h2):
#    continue
def sliced_wasserstein_clustering(r_S, K, projection_vectors, epsilon, h1, h2):
    #TODO Implement the original Sliced Wasserstein clustering function as described in Luan et al. (2025) "Automated regime classification in multidimensional time series data using sliced Wasserstein k-mean clustering"
    return 0

# (2) Sliced Wasserstein function with Monte Carlo (UnifOrtho) 
def sliced_wasserstein_clustering_unifortho(S, K, L, epsilon, h1, h2):
    '''
    INPUT : 
    - N_S : number of simulations to run with different random cluster initializations
    - S : (N*d) array of N samples in d dimensions
    - K : number of clusters (regimes)
    - L : number of projections 
    - epsilon : converge tolerance 
    - h1 : window size for lifting transfomation 
    - h2: sliding window offset parameter

    OUTPUT :
    - cluster_k_distributions : list of length K, each element is a list of projected distributions assigned to that cluster
    - centroids : list of length K, each element is a list of L ProjectedDistribution objects representing the centroid of that cluster
    - labels : array of length M (number of lifted samples), containing the cluster assignment for each lifted sample 
    '''


    r_S = np.diff(np.log(S), axis=0)
    N = r_S.shape[0]
    M = math.floor((N-(h1-h2))/h2)
    projected_emp_dist = unifortho_projection_vectors_opt(S, K, L, h1, h2) # Step 1 and 2: Projection and k-mean iteration (cache the projected distributions for all lifted samples)
    # Step 3: Sliced Wasserstein k-means clustering with convergence loop
    return sliced_wasserstein_clustering_conv_loop_opt(projected_emp_dist, K, M, L, epsilon)


def max_acc_unifortho_sim(N_S, S, r_s_true_regime, K, L, epsilon, h1, h2, test = False):
    r_S = np.diff(np.log(S), axis=0)
    N = r_S.shape[0]
    # perform lifting transfo on returns
    M = math.floor((N-(h1-h2))/h2)
    #start = time.perf_counter()
    projected_emp_dist = unifortho_projection_vectors_opt(S, K, L, h1, h2)
    #end = time.perf_counter()
    #print("Non_Optimized_Projection_Vectors Time", end- start)
    best_accuracy = 0
    #best_projected_emp_dist = None
    best_centroids = None
    best_labels = None
    for sim in range(N_S):
        _, centroids, labels = sliced_wasserstein_clustering_conv_loop_opt(projected_emp_dist, K, M, L, epsilon)
        print(f"Simulation {sim + 1}/{N_S} completed. Evaluating accuracy...")
        accuracy = mt.total_accuracy(S, r_s_true_regime, labels, h1, h2)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_centroids = centroids
            best_labels = labels
    
    ## TESTING FOR METRICS TO DIFFERENTIAL BULL/BEAR REGIMES 
    if (test): 
        for k in range(K):
            # Assuming your 'projected' object has a way to get the raw array of returns/atoms
            # e.g., projected.get_atoms() or projected.samples
            
            means = []
            variances = []
            skews = []
            cvars = []
            
            for projected in best_centroids[k]:
                sorted_atoms = projected.return_sorted_atoms() 

                means.append(np.mean(sorted_atoms))
                variances.append(np.var(sorted_atoms))
                skews.append(stats.skew(sorted_atoms))
                
                # 5% Expected Shortfall (Conditional VaR)
                cvars.append(np.mean(sorted_atoms[:max(1, int(len(sorted_atoms)*0.05))])) 

            print(f'--- Regime {k} ---')
            print(f'Mean: {np.mean(means)}')
            print(f'Variance: {np.mean(variances)}')
            print(f'Skewness: {np.mean(skews)}')
            print(f'CVaR (5%): {np.mean(cvars)}')
            print('-'*20)

    return projected_emp_dist, best_centroids, best_labels

def max_mccd_unifortho_sim(N_S, S, K, L, epsilon, h1, h2, metric = "CVaR"):
    #TODO for real time series do multiple runs and take the max mccd as the final output
    r_S = np.diff(np.log(S), axis=0)
    N = r_S.shape[0]
    # perform lifting transfo on returns
    M = math.floor((N-(h1-h2))/h2)
    #start = time.perf_counter()
    projected_emp_dist = unifortho_projection_vectors_opt(S, K, L, h1, h2)
    #end = time.perf_counter()
    #print("Non_Optimized_Projection_Vectors Time", end- start)
    best_mccd = 0
    #best_projected_emp_dist = None
    best_centroids = None
    best_labels = None
    for sim in range(N_S):
        _, centroids, labels = sliced_wasserstein_clustering_conv_loop_opt(projected_emp_dist, K, M, L, epsilon)
        print(f"Simulation {sim + 1}/{N_S} completed. Evaluating accuracy...")
        mccd = mt.mean_centroid_centroid_distance(centroids, K, p=2)
        if mccd > best_mccd:
            best_mccd = mccd
            best_centroids = centroids
            best_labels = labels
    
    #associate label 0 to bearish associate label 1 to bullish
    new_best_centroids, new_best_labels = choose_label(best_centroids, best_labels, metric, K)
    return projected_emp_dist, new_best_centroids, new_best_labels

def choose_label(best_centroids, best_labels,metric, K):
    '''
    INPUT: 
        - best_centroids : the output of the unifortho simulation 
        - best_labels : the output of the unifortho simulation 
        - metric : "CVaR" or "MeanVar"
    '''

    ## TESTING FOR METRICS TO DIFFERENTIAL BULL/BEAR REGIMES 
    dic_mean_var = {}
    dic_mean_cvar = {} 
    for k in range(K):
        # Assuming your 'projected' object has a way to get the raw array of returns/atoms
        # e.g., projected.get_atoms() or projected.samples    
        means = []
        variances = []
        skews = []
        cvars = []

        for projected in best_centroids[k]:
            sorted_atoms = projected.return_sorted_atoms() 

            means.append(np.mean(sorted_atoms))
            variances.append(np.var(sorted_atoms))
            skews.append(stats.skew(sorted_atoms))
        
            # 5% Expected Shortfall (Conditional VaR)
            cvars.append(np.mean(sorted_atoms[:max(1, int(len(sorted_atoms)*0.05))])) 

        avg = np.mean(means)
        var = np.mean(variances)
        skew = np.mean(skews) 
        cvar = np.mean(cvars)

            
        #print(f'--- Regime {k} ---')
        #print(f'Mean: {avg}')
        #print(f'Variance: {var}')
        #print(f'Skewness: {skew}')
        #print(f'CVaR (5%): {cvar}')
        #print('-'*20)

        dic_mean_var[k] = (avg, var)
        dic_mean_cvar[k] = cvar

    if (K == 2):
        if metric == "CVaR":
            # Associate 0 to the cluster with the min CVaR and 1 the cluster with the max CVaR
            if dic_mean_cvar[0] > dic_mean_cvar[1]:
                # Cluster 1 has the lower (worse) CVaR. We need to swap them.
                best_centroids[0], best_centroids[1] = best_centroids[1], best_centroids[0]
                best_labels = 1 - best_labels # Efficient numpy trick to swap 0s and 1s
                #print("Labels and Centroids swapped based on CVaR metric (0 is now bearish).")
            #else:
                #print("Labels kept as is based on CVaR metric (0 is already bearish).")

        elif metric == "MeanVar":
            # Associate 0 to the cluster with max variance and min mean, 1 to the other 
            mean0, var0 = dic_mean_var[0]
            mean1, var1 = dic_mean_var[1]

            if var1 > var0 and mean1 < mean0:
                # Cluster 1 is clearly the bearish one. Swap them.
                best_centroids[0], best_centroids[1] = best_centroids[1], best_centroids[0]
                best_labels = 1 - best_labels
                #print("Labels and Centroids swapped based on MeanVar metric (0 is now bearish).")
            elif var0 > var1 and mean0 < mean1:
                # Cluster 0 is clearly the bearish one. Keep as is.
                print("Labels kept as is based on MeanVar metric (0 is already bearish).")
            else:
                # Condition not strictly satisfied (e.g., one has higher mean AND higher variance)
                print("MeanVar condition not strictly satisfied. Labels left unchanged.")
            
        else:
            print(f"Metric '{metric}' not yet implemented.")

    else:
        print("Automated metric swapping not yet implemented for K != 2.")

    return best_centroids, best_labels


def compute_implied_proba(projected_emp_dist, centroids, labels, tau=None, lookback=5, use_gradient=False, gradient_weight=0.3):
    """
    Computes implied regime probabilities for each lifted sample,
    with a focused "switch signal" for the most recent sample.
    
    OUTPUT:
    - proba_matrix: (M, K) array of regime probabilities per sample
    - switch_proba: scalar, probability that the latest sample is 
                    switching away from its assigned regime
    - transition_matrix: (K, K) empirical transition matrix from labels
    """
    M = len(projected_emp_dist)
    K = len(centroids)

    # --- Step 1: Compute SW distance from every sample to every centroid ---
    # Reuse your optimized structure: extract into arrays
    h1_len = len(projected_emp_dist[0][0].return_sorted_atoms())
    L = len(projected_emp_dist[0])

    X = np.empty((M, L, h1_len))
    for m in range(M):
        for l in range(L):
            X[m, l, :] = projected_emp_dist[m][l].return_sorted_atoms()

    C = np.empty((K, L, h1_len))
    for k in range(K):
        for l in range(L):
            C[k, l, :] = centroids[k][l].return_sorted_atoms()

    # Vectorized SW-2 distance: (M, K)
    diff_sq = (X[:, None, :, :] - C[None, :, :, :]) ** 2
    dist_matrix = np.mean(np.sqrt(np.mean(diff_sq, axis=3)), axis=2)

    # --- Step 2: Calibrate temperature if not provided ---
    if tau is None:
        # Use half the mean inter-centroid distance as default
        centroid_dists = []
        for i in range(K):
            for j in range(i + 1, K):
                d_ij = np.mean(np.sqrt(np.mean((C[i] - C[j])**2, axis=1)))
                centroid_dists.append(d_ij)
        tau = 0.5 * np.mean(centroid_dists) if centroid_dists else 1.0

    # --- Step 3: Softmax over negative distances ---
    neg_scaled = -dist_matrix / tau
    neg_scaled -= neg_scaled.max(axis=1, keepdims=True)  # numerical stability
    exp_vals = np.exp(neg_scaled)
    proba_matrix = exp_vals / exp_vals.sum(axis=1, keepdims=True)  # (M, K)

    # --- Step 4: Empirical transition matrix from label history ---
    transition_matrix = np.zeros((K, K))
    for m in range(M - 1):
        transition_matrix[labels[m], labels[m + 1]] += 1
    # Normalize rows
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    transition_matrix /= row_sums

    
    
    # --- Step 5: Bayesian combination for the latest sample ---
    current_regime = labels[-1]
    # Prior: transition probabilities from current regime
    prior = transition_matrix[current_regime]  # (K,)
    # Likelihood: distance-based softmax for the last sample
    likelihood = proba_matrix[-1]  # (K,)
    # Posterior (multiply and renormalize)
    posterior = prior * likelihood
    posterior /= (posterior.sum() + 1e-12)  # avoid division by zero


    if use_gradient and M >= lookback:
        # Slope of distance to current centroid over recent samples
        # Positive slope = drifting away from current regime = weakening
        recent_dist_current = dist_matrix[-lookback:, current_regime]
        slope_current = np.polyfit(np.arange(lookback), recent_dist_current, deg=1)[0]

        # Slope of distance to each alternative centroid
        # Negative slope = approaching that regime = strengthening
        gradient_signal = np.zeros(K)
        for k in range(K):
            recent_dist_k = dist_matrix[-lookback:, k]
            slope_k = np.polyfit(np.arange(lookback), recent_dist_k, deg=1)[0]
            # Flip sign: negative slope (approaching) -> high score
            gradient_signal[k] = -slope_k

        # Normalize into a probability-like vector via softmax
        gradient_signal -= gradient_signal.max()
        gradient_proba = np.exp(gradient_signal / tau)
        gradient_proba /= (gradient_proba.sum() + 1e-12)  # avoid division by zero

        # Blend: posterior = (1 - w) * bayesian_posterior + w * gradient_signal
        posterior = (1 - gradient_weight) * posterior + gradient_weight * gradient_proba
        posterior /= (posterior.sum() + 1e-12)  # avoid division by zero

    switch_proba = 1.0 - posterior[current_regime]

    return proba_matrix, switch_proba, transition_matrix, posterior

    

if __name__ == "__main__":
    # Example usage
    r_S = np.random.rand(100, 3) # 100 samples in 3 dimensions
    K = 3
    L = 6
    epsilon = 1e-6
    h1 = 10
    h2 = 9
   # _,_,_ = sliced_wasserstein_clustering_unifortho(r_S, K, L, epsilon, h1, h2)
    



 





