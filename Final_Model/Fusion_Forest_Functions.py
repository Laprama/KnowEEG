import numpy as np
# Import models
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def fit_fusion_forest(X_train_mode_1 , X_train_mode_2, y_train, num_trees = 100):
    '''
    This code trains a 'Fusion Forest' using two modes of data, X_train_mode_1 and X_train_mode_2.
    
    Inputs: 
    - X_train_mode_1 - Dataframe containing training data features from mode 1 
    - X_train_mode_2 - Dataframe containing training data features from mode 2
    - y_train - numpy array, train labels
    - num_trees - Integer, Parameter determining number of trees in the fusion forest - default is 100
    
    Outputs: 
    - trained_trees_list - list, consisting of trained Decision Trees
    - selected_feats_per_tree - list of tuples, records which features from mode 1 and mode 2 were used to train each tree in the forest
    '''
    
    #Training the trees 
    trained_trees_list = []
    selected_feats_per_tree = [] 

    for i in range(num_trees):
        # Sample indices have replace = True (same sample can be selected twice) ________________________________________
        num_samples = X_train_mode_1.shape[0]
        sample_indices = np.random.choice(num_samples, size= num_samples , replace=True)
        
        # Feature indices have replace = True (same feature can't be selected twice) _______________________________________________
        num_mode_1_feats = X_train_mode_1.shape[1]
        feature_mode_1_indices = np.random.choice(num_mode_1_feats, size = int(np.sqrt(num_mode_1_feats)), replace=False)
        X_selected_mode_1_feats = X_train_mode_1.iloc[sample_indices , feature_mode_1_indices]
        
        num_mode_2_feats = X_train_mode_2.shape[1]
        feature_mode_2_indices = np.random.choice(num_mode_2_feats, size = int(np.sqrt(num_mode_2_feats)), replace=False)
        X_selected_mode_2_feats = X_train_mode_2.iloc[sample_indices , feature_mode_2_indices]
        
        selected_feats_per_tree.append((feature_mode_1_indices, feature_mode_2_indices))
        
        # Define selected train features and train labels
        X_all_selected = pd.concat([X_selected_mode_1_feats.reset_index(drop = True) , X_selected_mode_2_feats.reset_index(drop = True)], axis = 1)
        y_selected = y_train[sample_indices]
        
        clf = DecisionTreeClassifier()
        clf.fit(X_all_selected, y_selected)
        trained_trees_list.append(clf)

    return (trained_trees_list , selected_feats_per_tree)


def predict_fusion_forest(X_test_mode_1, X_test_mode_2, trained_trees_list, selected_feats_per_tree):
    '''
    This code predicts on unseen data using a trained 'Fusion Forest'
    Inputs: 
    - X_test_mode_1 - Dataframe containing test data features from mode 1 
    - X_test_mode_2 - Dataframe containing test data features from mode 2
    - trained_trees_list - list, consisting of trained Decision Trees
    - selected_feats_per_tree - list of tuples, records which features from mode 1 and mode 2 were used to train each tree in the forest
    
    Outputs: 
    - y_preds_all - numpy array, contains prediction from all trees for each test sample 
      with shape (num_trees , num_test_samples).  <-- Now these are SOFT predictions:
      for binary classification this is P(y=1) from each tree.

    Note : At the moment this prediction function only supports binary classification, will extend at a later date to include multiclass classification. 
    '''
    num_trees = len(trained_trees_list)
    ## Test the fusion forest
    y_preds_per_tree = []
    for i in range(num_trees):
        X_selected_mode_1_feats = X_test_mode_1.iloc[: , selected_feats_per_tree[i][0]]
        X_selected_mode_2_feats = X_test_mode_2.iloc[: , selected_feats_per_tree[i][1]]
        
        # Define selected train features and train labels
        X_test_all_selected = pd.concat(
            [X_selected_mode_1_feats.reset_index(drop = True),
             X_selected_mode_2_feats.reset_index(drop = True)],
            axis = 1
        )
    
        clf = trained_trees_list[i]

        # Get per-class probabilities for each sample
        proba = clf.predict_proba(X_test_all_selected)  # shape: (n_samples, n_classes)

        # Assume binary classification; extract probability of positive class (label == 1).
        # This is robust to class order: find which column corresponds to class 1.
        # If class 1 is not present in this tree (rare with bootstraps), use zeros.
        if 1 in clf.classes_:
            pos_idx = int(np.where(clf.classes_ == 1)[0][0])
            y_pred_soft = proba[:, pos_idx]
        else:
            # Tree never saw class 1 during training; contribute 0 prob for class 1
            y_pred_soft = np.zeros(proba.shape[0], dtype=float)
        # --- END CHANGE ---

        y_preds_per_tree.append(y_pred_soft)

    y_preds_all = np.array(y_preds_per_tree)  # shape: (num_trees, num_test_samples)

    return y_preds_all


def predict_fusion_forest_deprecated(X_test_mode_1, X_test_mode_2, trained_trees_list, selected_feats_per_tree):
    '''
    This function is deprecated as it uses 'hard' predictions (binary 1 or 0) from individual trees instead of 'soft' (probability) predictions that include more information.
    This code predicts on unseen data using a trained 'Fusion Forest'
    Inputs: 
    - X_test_mode_1 - Dataframe containing test data features from mode 1 
    - X_test_mode_2 - Dataframe containing test data features from mode 2
    - trained_trees_list - list, consisting of trained Decision Trees
    - selected_feats_per_tree - list of tuples, records which features from mode 1 and mode 2 were used to train each tree in the forest
    
    Outputs: 
    - y_preds_all - numpy array, contains prediction from all trees for each test sample with shape (num_trees , num_test_samples)
    '''
    
    num_trees = len(trained_trees_list)
    ## Test the fusion forest
    y_preds_per_tree = []
    for i in range(num_trees):
        X_selected_mode_1_feats = X_test_mode_1.iloc[: , selected_feats_per_tree[i][0]]
        X_selected_mode_2_feats = X_test_mode_2.iloc[: , selected_feats_per_tree[i][1]]
        
        # Define selected train features and train labels
        X_test_all_selected = pd.concat([X_selected_mode_1_feats.reset_index(drop = True) , X_selected_mode_2_feats.reset_index(drop = True)], axis = 1)
    
        clf = trained_trees_list[i]
        y_pred = clf.predict(X_test_all_selected)
        y_preds_per_tree.append(y_pred)

    y_preds_all = np.array(y_preds_per_tree)

    return y_preds_all