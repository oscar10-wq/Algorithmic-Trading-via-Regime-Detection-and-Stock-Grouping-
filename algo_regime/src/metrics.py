import numpy as np 
import math
import algo_regime.src.sWkmean as ws 
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix



def convert_prediction(N, labels, h1, h2):
    '''
    Convert the predicted regimes into a format compatible with the true regimes.
    We determine the overall label (regime) for point in the time series via a simple majority voting mechanism.
    '''
    #TODO Implement majority regime voting for partition of the time series 
    #as done with lifting transformation in the Wasserstein clustering function
    M = math.floor((N-(h1-h2))/h2)
    regime_pred = np.zeros((N, 1)) # Create an array to hold the original data and the predicted regimes
    for m in range(1, M+1): 
        start_idx = h2 * (m-1)
        end_idx = start_idx + h1-1
        regime_pred[start_idx:end_idx+1, 0] = labels[m-1] # Assign the predicted regime to the corresponding time points
    return regime_pred

def visualize_clustering_results(t, S, labels, h1, h2, K, returns = True): # K 2 or 3
    '''
    Visualize the clustering results by plotting the time series colored with prediction 
    '''
    r_S = np.diff(np.log(S), axis=0)
    N , d = r_S.shape
    M = math.floor((N-(h1-h2))/h2)
    regime_pred = np.zeros((N, 1)) # Create an array to hold the original data and the predicted regimes
    for m in range(1, M+1): 
        start_idx = h2 * (m-1)
        end_idx = start_idx + h1-1
        regime_pred[start_idx:end_idx+1, 0] = labels[m-1] # Assign the predicted regime to the corresponding time points

    fig , ax = plt.subplots(d,1, figsize=(18, 6))
    for i in range(d):
        if returns:
            ax[i].plot(t[1:], r_S[:, i], color='blue', label='Original Returns')
        else:
            ax[i].plot(t, S.iloc[:,i], color="blue", label ="Original Prices")
        for m in range(1, M+1): 
            start_idx = h2 * (m-1)
            end_idx = start_idx + h1-1
            if K == 2:
                if labels[m-1] == 1:
                    #print(f"Regime 1 predicted for time points {start_idx} to {end_idx}")
                    ax[i].axvspan(t[start_idx], t[end_idx+1], color='green', alpha=0.1) # Highlight the predicted regime with a red shaded area
                else:
                    #print(f"Regime 0 predicted for time points {start_idx} to {end_idx}")
                    ax[i].axvspan(t[start_idx], t[end_idx+1], color='red', alpha=0.1) # Highlight the predicted regime with a green shaded area
            if K == 3:
                if labels[m-1] == 0:
                    ax[i].axvspan(t[start_idx], t[end_idx+1], color='red', alpha=0.1) # Highlight the predicted regime with a red shaded area
                elif labels[m-1] == 1:
                    ax[i].axvspan(t[start_idx], t[end_idx+1], color='green', alpha=0.1) # Highlight the predicted regime with a green shaded area
                else:
                    ax[i].axvspan(t[start_idx], t[end_idx+1], color='orange', alpha=0.1) # Highlight the predicted regime with an orange shaded area
        if i ==0: 
            ax[i].plot([], [], color='red', alpha=0.3, label='Predicted Regime 0') # Add legend entry for regime 1
            ax[i].plot([], [], color='green', alpha=0.3, label='Predicted Regime 1') # Add legend entry for regime 0
            if K == 3:
                ax[i].plot([], [], color='orange', alpha=0.3, label='Predicted Regime 2') # Add legend entry for regime 2
            ax[i].legend(loc='upper right')
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.title('Clustering Results with Predicted Regimes')
    plt.legend()
    plt.tight_layout()
    plt.show()



def total_accuracy(S, true_regimes, labels, h1, h2):
    """
    Computes TA by handling the label switching problem and majority voting.
    
    true_regimes: np.array (N, 1) - The ground truth from your data generator
    labels: np.array (M,) - The labels from sWk-means
    """
    r_S = np.diff(np.log(S), axis=0)
    N,d = r_S.shape
    N = len(true_regimes)
    M = len(labels)
    
    # --- Step 1: Majority Voting (Convert Window Labels to Point Labels) ---
    # We count how many times each point is assigned to each label
    votes = np.zeros((N, len(np.unique(labels))))
    
    for m in range(M):
        start_idx = h2 * m
        end_idx = start_idx + h1
        # Add a 'vote' for the label assigned to this window
        votes[start_idx:end_idx, labels[m]] += 1

    # Final prediction for each point is the label with the most votes
    pred_regimes = np.argmax(votes, axis=1)

    # --- Step 2: Handle Label Switching (Permutations) ---
    # Find all unique labels (e.g., [0, 1])
    unique_labels = np.unique(pred_regimes)
    all_perms = list(permutations(unique_labels))
    
    max_acc = 0
    best_perm = None
    
    for perm in all_perms:
        # Map the labels based on the current permutation
        # e.g., if perm is (1, 0), then 0 becomes 1 and 1 becomes 0
        map_dict = dict(zip(unique_labels, perm))
        mapped_preds = np.array([map_dict[p] for p in pred_regimes])

        # Calculate accuracy for this specific mapping
        current_acc = sum(mapped_preds.flatten() == true_regimes.flatten()) / (sum(mapped_preds.flatten() == true_regimes.flatten()) + sum(mapped_preds.flatten() != true_regimes.flatten()))
        
        if current_acc > max_acc:
            max_acc = current_acc
            best_perm = perm
            
    return max_acc


def mean_squared_point_centroid_distance(centroids, labels, projected_emp_dist, K, p=2):
    """
    Computes the Mean Squared Point-to-Centroid Distance exactly as defined in the paper.
    """
    total_mspd = 0.0
    # Outer sum over the K clusters
    for k in range(K):
        # Find indices of all distributions assigned to cluster k
        indices_in_k = np.where(labels == k)[0]

        #TODO Check that the norm || C_k || is indeed the below formula 
        card_C_k = len(indices_in_k)

        # Safety check: if cluster is empty, skip to avoid division by zero
        if card_C_k == 0:
            continue 


        centroid_k = centroids[k]
        cluster_sum = 0.0
        
        # Inner sum: over all distributions mu in C_k
        for idx in indices_in_k:
            mu_j = projected_emp_dist[idx]
            
            # Compute Sliced Wasserstein distance
            sw_dist_obj = ws.sliced_wasserstein_distance(mu_j, centroid_k, p=p)
            dist = sw_dist_obj.compute_distance_matrix()
            
            # Square the distance
            cluster_sum += (dist ** p)
            
        # Divide by |C_k| (completing the inner part of the formula)
        cluster_variance = cluster_sum / card_C_k
        
        # Add to the outer sum
        total_mspd += cluster_variance
        
    # Finally, multiply by 1/K
    return total_mspd / K

def mean_centroid_centroid_distance(centroids, K, p=2): 
    """
    Computes the Mean Centroid-Centroid Distance (MCCD).
    Represents the average Sliced Wasserstein distance between all unique pairs of cluster centroids.
    """
    # Safety check: if there is only 1 cluster, the distance between centroids is 0
    if K < 2:
        return 0.0
        
    num_combinations = math.comb(K, 2)
    total_mccd = 0.0 
    
    # Iterate ONLY over unique pairs (k < j) to avoid double counting and save computation time
    for k in range(K):
        for j in range(k + 1, K): 
            # Note: Used the dynamic 'p' parameter instead of hardcoding p=2
            sw_dist_obj = ws.sliced_wasserstein_distance(centroids[k], centroids[j], p=p)
            total_mccd += sw_dist_obj.compute_distance_matrix()
            
    # Divide by the number of unique pairs to get the true mean
    return total_mccd / num_combinations


def balanced_accuracy(S, true_regimes, labels, h1, h2):
    #TODO # test this function 

    """
    Compute Balanced Accuracy 
    """
    r_S = np.diff(np.log(S), axis=0)
    N,d = r_S.shape
    N = len(true_regimes)
    M = len(labels)
    
    # --- Step 1: Majority Voting (Convert Window Labels to Point Labels) ---
    # We count how many times each point is assigned to each label
    votes = np.zeros((N, len(np.unique(labels))))
    
    for m in range(M):
        start_idx = h2 * m
        end_idx = start_idx + h1
        # Add a 'vote' for the label assigned to this window
        votes[start_idx:end_idx, labels[m]] += 1

    # Final prediction for each point is the label with the most votes
    pred_regimes = np.argmax(votes, axis=1)

    # --- Step 2: Handle Label Switching (Permutations) ---
    # Find all unique labels (e.g., [0, 1])
    unique_labels = np.unique(pred_regimes)
    all_perms = list(permutations(unique_labels))
    
    max_acc = 0
    best_perm = None
    
    for perm in all_perms:
        # Map the labels based on the current permutation
        # e.g., if perm is (1, 0), then 0 becomes 1 and 1 becomes 0
        map_dict = dict(zip(unique_labels, perm))
        mapped_preds = np.array([map_dict[p] for p in pred_regimes])

        # Calculate accuracy for this specific mapping
        current_acc = sum(mapped_preds.flatten() == true_regimes.flatten()) / (sum(mapped_preds.flatten() == true_regimes.flatten()) + sum(mapped_preds.flatten() != true_regimes.flatten()))
        
        if current_acc > max_acc:
            max_acc = current_acc
            best_perm = mapped_preds
    return balanced_accuracy_score(true_regimes, best_perm)


def confusion_matrix_WS(S, true_regimes, labels, h1, h2):
    #TODO # test this function 
    """
    Compute Confusion Matrix 
    """
    r_S = np.diff(np.log(S), axis=0)
    N,d = r_S.shape
    N = len(true_regimes)
    M = len(labels)
    
    # --- Step 1: Majority Voting (Convert Window Labels to Point Labels) ---
    # We count how many times each point is assigned to each label
    votes = np.zeros((N, len(np.unique(labels))))
    
    for m in range(M):
        start_idx = h2 * m
        end_idx = start_idx + h1
        # Add a 'vote' for the label assigned to this window
        votes[start_idx:end_idx, labels[m]] += 1

    # Final prediction for each point is the label with the most votes
    pred_regimes = np.argmax(votes, axis=1)

    # --- Step 2: Handle Label Switching (Permutations) ---
    # Find all unique labels (e.g., [0, 1])
    unique_labels = np.unique(pred_regimes)
    all_perms = list(permutations(unique_labels))
    
    max_acc = 0
    best_perm = None
    
    for perm in all_perms:
        # Map the labels based on the current permutation
        # e.g., if perm is (1, 0), then 0 becomes 1 and 1 becomes 0
        map_dict = dict(zip(unique_labels, perm))
        mapped_preds = np.array([map_dict[p] for p in pred_regimes])

        # Calculate accuracy for this specific mapping
        current_acc = sum(mapped_preds.flatten() == true_regimes.flatten()) / (sum(mapped_preds.flatten() == true_regimes.flatten()) + sum(mapped_preds.flatten() != true_regimes.flatten()))
        
        if current_acc > max_acc:
            max_acc = current_acc
            best_perm = mapped_preds
    return confusion_matrix(true_regimes, best_perm)

def simulate_clustering_data(S, true_regimes, K, L, epsilon, h1, h2, totalal_acc=True):
    '''
    Simulate for multiple hyperparameter using the original clustering algo
    '''
    # TODO
    return 0


def simulate_unifortho_data(N_S, S,true_regimes, K, L, epsilon, h1, h2, total_acc=True):
        '''
        INPUT : 
        - N_S : number of simulations 
        - S : original time series data (N, d)
        - true_regimes : ground truth regimes (N-1,1) for returns of S
        - K : number of regimes 
        - L : number of projections
        - epsilon : converge tolerance
        - h1 : window size for lifting transfomation
        - h2: sliding window offset parameter

        OUTPUT :
        - Maximum of total accuracy accross N_S simulations
        - Median of total accuracy accross N_S simulations
        '''

        r_S = np.diff(np.log(S), axis=0)
        N = r_S.shape[0]
        M = math.floor((N-(h1-h2))/h2)
        ### NEW PART ### 
        projected_emp_dist = ws.unifortho_projection_vectors_opt(S, K, L, h1, h2) # Step 1 and 2: Projection and k-mean iteration (cache the projected distributions for all lifted samples)
        accuracies = []
        mcc_distances = []
        for i in range(N_S):
            projected_emp_dist, centroids, labels = ws.sliced_wasserstein_clustering_conv_loop_opt(projected_emp_dist,K,M, L, epsilon)
            print(f"Simulation {i + 1}/{N_S} completed. Evaluating Performance Metrics...")
            acc = total_accuracy(S, true_regimes, labels, h1, h2)
            mccd = mean_centroid_centroid_distance(centroids, K, p=2)
            accuracies.append(acc)
            mcc_distances.append(mccd)
        index_max_mccd = np.argmax(np.array(mcc_distances))
        best_unsupervised_accuracy = accuracies[index_max_mccd]
        return np.max(accuracies), np.median(accuracies), best_unsupervised_accuracy


def display_results(N_C,results, liste_L, liste_h1_h2, types):
    #Display the results in a clear and concise manner, such as a table or a plot.
    #TODO Add the mean squared point to point distance metric to the results and display it in the table as well

    df = pd.DataFrame(results, columns=['h1', 'h2', 'L', f'Max (Type {types[0]})', f'Max (Type {types[1]})', f'Median (Type {types[0]})', f'Median (Type {types[1]})', f'Max(MCCD) ({types[0]})',f'Max(MCCD) ({types[1]})'])
    # 1. Prepare Data
    plot_df = df.round(4)
    num_rows = len(plot_df)
    num_cols = len(plot_df.columns)

    # 2. Create Figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # 3. Create the Table
    the_table = ax.table(cellText=plot_df.values, 
                        colLabels=plot_df.columns, 
                        cellLoc='center', 
                        loc='center')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.2, 3.5)

    # 4. Styling, Heatmap, and Advanced Merging Logic
    cmap = cm.get_cmap('Reds')
    acc_columns = [3, 4, 5, 6, 7, 8] # Accuracy column indices
    print(acc_columns)

    # Identify groups for h1, h2
    # We assume the dataframe is sorted by h1, h2
    group_size = len(liste_L) 

    for row in range(num_rows + 1): # +1 for header
        for col in range(num_cols):
            cell = the_table[row, col]
            
            if row == 0: # Header
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f2f2f2')
                continue

            # --- HEATMAP LOGIC ---
            if col in acc_columns:
                val = float(plot_df.iloc[row-1, col])
                cell.set_facecolor(cmap(val))
                if val > 0.7: cell.get_text().set_color('white')
            
            # --- BORDER & CENTERING LOGIC (The "Erase Lines" part) ---
            if col in [0, 1]:
                # Calculate position within the group (0 to group_size-1)
                pos_in_group = (row - 1) % group_size
                
                # Change background to light grey to distinguish parameters
                cell.set_facecolor('#f9f9f9')
                
                # Erase horizontal lines
                if pos_in_group == 0:
                    # Top of group: keep top, erase bottom
                    cell.visible_edges = 'RLT' 
                elif pos_in_group == group_size - 1:
                    # Bottom of group: keep bottom, erase top
                    cell.visible_edges = 'RLB'
                else:
                    # Middle of group: erase top and bottom
                    cell.visible_edges = 'RL'
                
                # Centering the text: Only show text in the "middle" row of the group
                middle_idx = group_size // 2
                if pos_in_group != middle_idx:
                    cell.get_text().set_text("")
                else:
                    cell.set_text_props(weight='bold')

    #plt.title(f'Accuracy of UnifOrtho Sliced Wasserstein:  Hyperparameter Groups for {N_C} Simulations', pad=30, fontsize=16, weight='bold')
    plt.show()
    return 
    
