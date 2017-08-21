from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math

 # auto-detect width of terminal for displaying dataframes
pd.set_option('display.max_columns',0)
pd.options.mode.chained_assignment = None 
np.set_printoptions(suppress=True)


'''                          #######################
#-------------------------   ## --- Functions --- ##     -----------------------
                             #######################
'''



def random_forest_validation(path, train_labels, num_trees):
    ''' Train the model and validate.
        Also returns the accuracy with a F1 score.

    Args:
        path: str; main path
        train_labels: pandas data frame; contains feature set and labels 
        num_trees: int; number of trees in the random forest

    Returns: 
        F_score:    float; accuracy model
        validation: pandas data frame; predicted products, true labels, 
                    and predicted labels
        
    '''
    # Shuffle data before training model by pulling all user IDs and shuffling
    # Subset into validation sets by creating new data frames from shuffled users IDs
    users = train_labels['user_id'].unique()
    np.random.shuffle(users)
    split = int(0.8*len(users)) # split into 80% validation, 20% test
    training_users = users[:split]
    # Create a column in original data frame that matches users to right data set
    train_labels['TF'] = train_labels['user_id'].isin(training_users)
    
    train_set = train_labels[train_labels['TF']==True]
    train_set = train_set.drop('TF', axis=1)
    
    validation_set = train_labels[train_labels['TF']==False]
    validation_set = validation_set.drop('TF', axis=1)
    
    train_set = train_set.as_matrix()
    validation_set = validation_set.as_matrix()
    
    # Split into the samples and labels from the main data sets
    # Ignore first two columns because they contain user ID information
    train_samples = train_set[:, 2:-1]
    train_labels = train_set[:, -1]
    
    valid_samples = validation_set[:, 2:-1]
    valid_labels = validation_set[:, -1]
    
    valid_user_prod = validation_set[:, :2]
    valid_user_prod = valid_user_prod.astype(int)
    
    # Train the random forest
    clf = RandomForestClassifier(n_estimators=num_trees)
    clf = clf.fit(train_samples, train_labels)    
    
    # Predict the labels (1/0 reordered/not reordered product)
    prediction = clf.predict(valid_samples)
    
    # Check accuracy with F1 score!
    validation = {'test_labels': valid_labels,
        'predicted': prediction
        }
    validation = pd.DataFrame(validation)

    tmp = []
    for i in validation.index:
        if ((validation['test_labels'][i]==1) & \
            (validation['predicted'][i]==1)):
            tmp.append('TP')
        elif ((validation['test_labels'][i]==0) & \
            (validation['predicted'][i]==1)):
            tmp.append('FP')
        elif ((validation['test_labels'][i]==1) & \
            (validation['predicted'][i]==0)):
            tmp.append('FN')
        elif ((validation['test_labels'][i]==0) & \
            (validation['predicted'][i]==0)):
            tmp.append('TN')
        else:
            tmp.append('unknown')

    validation['accuracy'] = tmp

    accuracy_values = validation['accuracy'].value_counts().reset_index()
    true_neg = accuracy_values.loc[accuracy_values['index'] == 'TN', 'accuracy'].item()
    false_neg = accuracy_values.loc[accuracy_values['index'] == 'FN', 'accuracy'].item()
    false_pos = accuracy_values.loc[accuracy_values['index'] == 'FP', 'accuracy'].item()
    true_pos = accuracy_values.loc[accuracy_values['index'] == 'TP', 'accuracy'].item()

    precision = true_pos/(true_pos+false_pos)
    recall = true_pos / (true_pos+false_neg)
    F_score = 2 * (precision*recall)/(precision+recall)
    print('F_score = ', F_score)
    
    validation['user_id'] = valid_user_prod[:,0]
    validation['product_id'] = valid_user_prod[:,1]
    
    return F_score, validation


def k_fold(samples, k, forest_sizes):
    ''' Cross validate the random forest to find optimal number of trees

    Args:
        samples:        pandas data frame; contains feature set and labels 
        num_trees:      int; number of trees in the random forest
        k:              int; number of folds
        forest_sizes:   list of int; forest sizes to test

    Returns: 
        results:        A np.array forest_sizes x k large containing the 
                        F1 scores associated with each forest size
        
    '''
    
    # Shuffle data before training model by pulling all user IDs and shuffling
    # Subset into validation sets by creating new data frames from shuffled users IDs
    users = samples['user_id'].unique()
    np.random.shuffle(users)
    
    # Creates empty array k x forest_sizes large 
    results_rows = len(forest_sizes)    
    results=np.zeros(shape=(results_rows,k+1), dtype=np.float)  
    
    for num_trees in forest_sizes:
        # Finds the index of the num_trees for correct row index   
        tree_position = forest_sizes.index(num_trees)
        # Label matrix row with num_tree being tested
        results[tree_position,0]= num_trees
        
        # Subset the data for k fold validation, find size of validation set
        num_rows = len(users)
        row_block = (num_rows//k) # eg 500 samples, 5-fold = 100 valid, 400 train
        
        # Run k-times for cross validation. 
        # Separate out each validation group per loop
        for validationgroup in range(0, k):
                # Find proper start/end for each validation set
                indexFirst= row_block * validationgroup     
                indexSecond = row_block * (validationgroup+1)
                
                # split all the data into validation and training blocks
                training_users = users[indexFirst:indexSecond]
                samples['TF'] = samples['user_id'].isin(training_users)
                
                train_set = samples[samples['TF']==True]
                train_set = train_set.drop('TF', axis=1)
    
                validation_set = samples[samples['TF']==False]
                validation_set = validation_set.drop('TF', axis=1)
    
                train_set = train_set.as_matrix()
                validation_set = validation_set.as_matrix()
    
                trainingdata = train_set[:, 2:-1]
                traininglabels = train_set[:, -1]
    
                validationdata = validation_set[:, 2:-1]
                validationlabels = validation_set[:, -1]
            
                # Append results of the validation to results
                results[tree_position, validationgroup+1] = \
                    k_fold_forest(
                        trainingdata, traininglabels, validationdata,
                        validationlabels, num_trees, validationgroup
                    )
    print (results)
    return results
    
    
def k_fold_forest(trainingdata, traininglabels, validationdata, validationlabels, num_trees, validationgroup):
    ''' Random forest used during k-fold validation.

    Args:
        trainingdata:       matrix; training samples
        traininglabels:     matrix; training labels
        validationdata:     matrix; validaiton samples
        validationlabels:   matrix; validation labels
        num_trees:          int; size of forest for model
        validationgroup:    int; which k-fold number is being tested

    Returns: 
        F-score:            accuracy of model
        
    '''
    
    # Train the forest
    clf = RandomForestClassifier(n_estimators=num_trees)
    clf = clf.fit(trainingdata, traininglabels)        
    prediction = clf.predict(validationdata)

    
    # Check accuracy with F1 score!
    validation = {'test_labels': validationlabels,
        'predicted': prediction
        }
    validation = pd.DataFrame(validation)

    tmp = []
    for i in validation.index:
        if ((validation['test_labels'][i]==1) & \
            (validation['predicted'][i]==1)):
            tmp.append('TP')
        elif ((validation['test_labels'][i]==0) & \
            (validation['predicted'][i]==1)):
            tmp.append('FP')
        elif ((validation['test_labels'][i]==1) & \
            (validation['predicted'][i]==0)):
            tmp.append('FN')
        elif ((validation['test_labels'][i]==0) & \
            (validation['predicted'][i]==0)):
            tmp.append('TN')
        else:
            tmp.append('unknown')

    validation['accuracy'] = tmp

    accuracy_values = validation['accuracy'].value_counts().reset_index()
    true_neg = accuracy_values.loc[accuracy_values['index'] == 'TN', 'accuracy'].item()
    false_neg = accuracy_values.loc[accuracy_values['index'] == 'FN', 'accuracy'].item()
    false_pos = accuracy_values.loc[accuracy_values['index'] == 'FP', 'accuracy'].item()
    true_pos = accuracy_values.loc[accuracy_values['index'] == 'TP', 'accuracy'].item()

    precision = true_pos/(true_pos+false_pos)
    recall = true_pos / (true_pos+false_neg)
    F_score = 2 * (precision*recall)/(precision+recall)
    
    #in case you want to print out results as everything runs.
    print ("forest size = ", num_trees, "Accuracy of test group ", 
        validationgroup, " = ", F_score, "\n") 
    return F_score 
    
def random_forest_submit(path, train_set, test_samples, num_trees):
    ''' Train model based on k-fold validation determined number of trees
        Predict labels for the test set for Kaggle competition

    Args:
        path:           str; main path
        train_set:      pandas data frame; contains training feature set and labels
        test_samples:   pandas data frame; the features only of the test samples
        num_trees:      int; number of trees in the random forest

    Returns: 
        prediction:    pandas data frame; predicted labels from the model
        
    '''
    
    train_set = train_set.as_matrix()
    test_samples = test_samples.as_matrix()
    
    np.random.shuffle(train_set)

    train_samples = train_set[:, 2:-1]
    train_labels = train_set[:, -1]
    
    # Save the main training set with all the labels
    user_prod = test_samples[:, :2]
    
    test_samples = test_samples[:, 2:]
    
    clf = RandomForestClassifier(n_estimators=num_trees)
    clf = clf.fit(train_samples, train_labels)    
    
    prediction = clf.predict(test_samples)
    
    # Create a pandas data frame for return
    prediction = pd.DataFrame(prediction)
    prediction.columns = ['predicted']
    prediction['user_id'] = user_prod[:, 0]
    prediction['product_id'] = user_prod[:, 1]

    return prediction

def save_submission(path, results, order_labels):
    ''' Modify format of predicted reordered products for submission to Kaggle.
        Save as a CSV for submission.
        user id, product number product number product number

    Args:
        path: str; main path
        order_labels: all order number IDs matched with user number IDs
    Returns: 
        None
        
    '''
    
    path = path + '/test/'
    
    results['product_id'] = results['product_id'].astype(int)
    # Match all users with their final order number     
    predicted_reordered = pd.merge(results, order_labels[
        ['user_id', 'order_id']], on='user_id')
    # Find unique product IDs  for each user  to prevent duplicates being listed
    predicted_reordered = predicted_reordered.groupby(
        ['order_id', 'predicted']
        )['product_id'].unique().reset_index()
    predicted_reordered = predicted_reordered.drop_duplicates(
        'order_id',keep='last'
        ).sort_values('order_id')
    # Sort out the users who had reordered products
    reordered = predicted_reordered[
        predicted_reordered['predicted']==1.0
        ].sort_values('order_id')
    # Also find users who did not reorder any products
    noreorders = predicted_reordered[
        predicted_reordered['predicted']==0.0
        ].sort_values('order_id')
    
    # Create a list format matching Kaggle Submission example file
    # user id, product number product number
    reordered = reordered.drop('predicted', axis=1)
    reordered['product_id'] = reordered['product_id'].apply(
        lambda x: " ".join(str(i) for i in x))
    # If user predicted to not reorder products, assign NaN to product_id
    noreorders = noreorders.drop('predicted', axis=1)
    noreorders['product_id'] = math.nan

    submission = pd.concat(
        [reordered,noreorders], axis=0
        ).sort_values('order_id')
    submission.columns = ['order_id','products']
    submission.to_csv(path+'submission.csv', sep=',', index=False)
    
    
    return

'''                       #######################
#----------------------   ## ---    MAIN   --- ##     --------------------------
                          #######################
'''
if __name__ == '__main__':  
    path = '/Users/judyjinn/Python/CDIPS/instacart'
    
    # train_labels = pd.read_csv(path+'/training/PriorsInTrain_withIDs.csv')
    train_labels = pd.read_csv(path+'/training/5000_user_prod_withIDs.csv')
    order_labels = pd.read_csv(path+'/instacart_2017_05_01/orders_train.csv')
    

    # K-fold cross validation
    k = 5
    forest_sizes = [5,6,7,8]

    k_fold_results = []
    k_fold_results.append(k_fold(train_labels, k, forest_sizes))

    #the previous script saves the array as a list, this picks out the array
    k_results = k_fold_results[0]

    k_results = pd.DataFrame(
        data = k_results[:,1:],
        index = list(map(int,k_results[:,0])),
        columns = list(map(str,list(range(0,k))))
        )

    k_results['avg acc'] = k_results.mean(axis=1)

    print(k_results)

    # # Submit to Kaggle
    num_trees = 5
    orders_test = pd.read_csv(path+'/instacart_2017_05_01/orders_test.csv')
    train_set = pd.read_csv(path+'/training/PriorsInTrain_withIDs.csv')
    test_samples = pd.read_csv(path+'/test/PriorsInTest_4testing_user_prod_withIDs.csv')

    prediction = random_forest_submit(path, train_set, test_samples, num_trees)
    save_submission(path, prediction, orders_test)
   
   
  