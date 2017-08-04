from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

 # auto-detect width of terminal for displaying dataframes
pd.set_option('display.max_columns',0)
pd.options.mode.chained_assignment = None 


'''                          #######################
#-------------------------   ## --- Functions --- ##     -----------------------
                             #######################
'''

def tree_validation(path, train_labels):
    
    
    # Split into subsets for Validation
    tr_grp_size = 0.8
    users = train_labels['user_id'].unique()
    num_users = train_labels['user_id'].nunique()
    user_split = users[int(num_users * tr_grp_size)]
    
    train_labels['user_id'] = train_labels['user_id'].astype(int)
    
    train_set = train_labels[train_labels['user_id'] < user_split]
    validation_set = train_labels[train_labels['user_id'] >= user_split]
    
    train_samples = train_set.drop(['user_id', 'product_id', 'reordered'], axis=1).as_matrix()
    train_labels = train_set['reordered'].as_matrix()
    valid_samples = validation_set.drop(['user_id', 'product_id', 'reordered'], axis=1).as_matrix()
    valid_labels = validation_set['reordered'].as_matrix()
    
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_samples, train_labels)
    
    
    prediction = clf.predict(valid_samples)
    
    
    
    # Check accuracy with F1 score!
    validation = validation_set[['user_id', 'product_id', 'reordered']]
    validation['predicted'] = prediction
    
    tmp = []
    for i in validation.index:
        if ((validation['reordered'][i]==1.0) & \
            (validation['predicted'][i]==1.0)):
            tmp.append('TP')
        elif ((validation['reordered'][i]==0.0) & \
            (validation['predicted'][i]==1.0)):
            tmp.append('FP')
        elif ((validation['reordered'][i]==1.0) & \
            (validation['predicted'][i]==0.0)):
            tmp.append('FN')
        elif ((validation['reordered'][i]==0.0) & \
            (validation['predicted'][i]==0.0)):
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
    print('Decision Tree F_score = ', F_score)
    
    return F_score

def tree_test(path, train_labels):
    train_set = pd.read_csv(path+'/training/PriorsInTrain_user_prod_withIDs.csv')
    test_samples = pd.read_csv(path+'/training/PriorsInTest_4testing_user_prod_withIDs.csv')
    
    train_samples = train_set.drop(['user_id', 'product_id', 'reordered'], axis=1).as_matrix()
    train_labels = train_set['reordered'].as_matrix()
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_samples, train_labels)
    
    prediction = clf.predict(valid_samples)
    
    return prediction


def k_fold_forest(train_samples, train_labels, valid_samples, valid_labels, grp_num):
    print('hiiii')
    return
    
def forest_train(path, train_labels):

    # Split into subsets for Validation
    tr_grp_size = 0.8
    users = train_labels['user_id'].unique()
    num_users = train_labels['user_id'].nunique()
    user_split = users[int(num_users * tr_grp_size)]

    train_labels['user_id'] = train_labels['user_id'].astype(int)

    train_set = train_labels[train_labels['user_id'] < user_split]
    validation_set = train_labels[train_labels['user_id'] >= user_split]

    train_samples = train_set.drop(['user_id', 'product_id', 'reordered'], axis=1).as_matrix()
    train_labels = train_set['reordered'].as_matrix()
    valid_samples = validation_set.drop(['user_id', 'product_id', 'reordered'], axis=1).as_matrix()
    valid_labels = validation_set['reordered'].as_matrix()
    
    
    clf = RandomForestClassifier(n_estimators=3)
    clf = clf.fit(train_samples, train_labels)
    
    
    prediction = clf.predict(valid_samples)
    
    
    
    # Check accuracy with F1 score!
    validation = validation_set[['user_id', 'product_id', 'reordered']]
    validation['predicted'] = prediction
    
    tmp = []
    for i in validation.index:
        if ((validation['reordered'][i]==1.0) & \
            (validation['predicted'][i]==1.0)):
            tmp.append('TP')
        elif ((validation['reordered'][i]==0.0) & \
            (validation['predicted'][i]==1.0)):
            tmp.append('FP')
        elif ((validation['reordered'][i]==1.0) & \
            (validation['predicted'][i]==0.0)):
            tmp.append('FN')
        elif ((validation['reordered'][i]==0.0) & \
            (validation['predicted'][i]==0.0)):
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
    print('Random Forest F_score = ', F_score)
    
    return F_score
      
def R_tree_validation():
    ##### R data

    R_train = pd.read_csv('/Users/judyjinn/Python/CDIPS/instacart/R_train.csv', sep=',')
    R_train_X = R_train.as_matrix()

    R_test = pd.read_csv('/Users/judyjinn/Python/CDIPS/instacart/R_test.csv', sep=',')
    R_test = R_test.drop('product_id')
    R_test_X = R_test.as_matrix()



    train_samples_70 = train_samples[:5932262, :-1]
    train_labels_70 = R_train_X[:5932262,-1]

    valid_samples_30 = train_samples[5932263:, :-1]
    valid_labels_30 = R_train_X[5932263:,-1]



    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_samples_70, train_labels_70)


    prediction = clf.predict(valid_samples_30)



    # Check accuracy with F1 score!
    validation = {'test_labels': valid_labels_30,
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

    return  F_score  




'''                       #######################
#----------------------   ## ---    MAIN   --- ##     --------------------------
                          #######################
'''
if __name__ == '__main__':  
    path = '/Users/judyjinn/Python/CDIPS/instacart'
    # train_labels = pd.read_csv(path+'/training/PriorsInTrain_user_prod_withIDs.csv')
    train_labels = pd.read_csv(path+'/training/5000_user_prod_withIDs.csv')
    
    # tree_validation(path, train_labels)
    # run_test()
    
    forest_train(path,train_labels)




