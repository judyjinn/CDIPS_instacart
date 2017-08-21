import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import math

# auto-detect width of terminal for displaying dataframes
pd.set_option('display.max_columns',0)
pd.options.mode.chained_assignment = None 
np.set_printoptions(suppress=True)


'''                          #######################
#-------------------------   ## --- Functions --- ##     -----------------------
                             #######################
'''
def open_concat_sets(path):
    path = path+'/instacart_2017_05_01'
    print('opening prior_concat full set')
    prior_concat = pd.read_csv(path+"/prior_concat.csv", dtype={
            'order_id': np.uint32, 
            'user_id': np.uint32, 
            'eval_set': 'category', 
            'order_number': np.uint8, 
            'order_dow': np.uint8, 
            'order_hour_of_day': np.uint8, 
            'days_since_prior_order': np.float32, 
            'product_id': np.uint32, 
            'add_to_cart_order': np.uint8, 
            'reordered': np.uint8,
            'aisle_id': np.int32, 
            'aisle': 'category',
            'department_id': np.int32, 
            'department': 'category'
            })
    print('opening train_concat full set')
    train_concat = pd.read_csv(path+"/train_concat.csv", dtype={
            'order_id': np.uint32, 
            'user_id': np.uint32, 
            'eval_set': 'category', 
            'order_number': np.uint8, 
            'order_dow': np.uint8, 
            'order_hour_of_day': np.uint8, 
            'days_since_prior_order': np.float32, 
            'product_id': np.uint32, 
            'add_to_cart_order': np.uint8, 
            'reordered': np.uint8,
            'aisle_id': np.int32, 
            'aisle': 'category',
            'department_id': np.int32, 
            'department': 'category'
            })
    print('opening orders_train set')
    orders_train = pd.read_csv(path+"/orders_train.csv", dtype={
            'order_id': np.uint32, 
            'user_id': np.uint32, 
            'eval_set': 'category', 
            'order_number': np.uint8, 
            'order_dow': np.uint8, 
            'order_hour_of_day': np.uint8, 
            'days_since_prior_order': np.float32
            })
            
    print('opening orders_train set')
    orders_test = pd.read_csv(path+"/orders_test.csv", dtype={
            'order_id': np.uint32, 
            'user_id': np.uint32, 
            'eval_set': 'category', 
            'order_number': np.uint8, 
            'order_dow': np.uint8, 
            'order_hour_of_day': np.uint8, 
            'days_since_prior_order': np.float32
            })
    print('finished opening')        
    

    return prior_concat, train_concat, orders_train, orders_test

def open_training_sets(path):
    path = path+'/training/subset'
    
    print('opening priors_in_train')
    priors_in_train = pd.read_csv(path+'/priors_in_train.csv',
        dtype={
            'order_id': np.uint32,
            'user_id': np.uint32,
            'order_number': np.uint8,
            'order_dow': np.uint8,
            'order_hour_of_day': np.uint8,
            'days_since_prior_order': np.float32,
            'product_id': np.uint32,
            'add_to_cart_order': np.uint8,
            'reordered': np.uint8,
            'aisle_id': np.int32,
            'department_id': np.int32,
        })
    print('opening orders_train')
    orders_train_info = pd.read_csv(path+'/orders_train_full.csv',
        dtype={
            'order_id': np.uint32,
            'user_id': np.uint32,
            'order_number': np.uint8,
            'order_dow': np.uint8,
            'order_hour_of_day': np.uint8,
            'days_since_prior_order': np.float32,
            'product_id': np.uint32,
            'add_to_cart_order': np.uint8,
            'reordered': np.uint8,
            'aisle_id': np.int32,
            'department_id': np.int32,
        })        
    print('finished opening')
    return priors_in_train, orders_train_info
    
def graph_histogram(path, data, save_name, bins):
    path = path+'/graphs/'
    
    fig = plt.figure()
    ax = fig.gca()
    plt.style.use('ggplot')
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    # plt.ticklabel_format(useOffset=False) # turn off scientific plotting
    fig.suptitle(save_name, fontsize = 30)

    ax.hist(data, bins, facecolor='green')

    ax.set_facecolor('white')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.subplots_adjust(bottom = None, top = 0.9)
    fig.savefig(path+save_name +'.png')
    plt.close()
    
    return
    
def graph_bar(path, x, y, x_labels, save_name, bins):
    path = path+'/graphs/'
    
    fig = plt.figure()
    ax = fig.gca()
    plt.style.use('ggplot')
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    # plt.ticklabel_format(useOffset=False) # turn off scientific plotting
    fig.suptitle(save_name, fontsize = 30)

    ax.bar(x, y, color='green')

    ax.set_facecolor('white')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xticks(x, x_labels)
    ax.autoscale(enable=True, axis='both', tight=None)
    
    plt.subplots_adjust(bottom = None, top = 0.9)
    fig.savefig(path+save_name +'.png')
    plt.close()
    
    return
    
def graph_scatter(path, x, y, save_name):
    path = path+'/graphs/'
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.gca()
    plt.style.use('ggplot')
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    # plt.ticklabel_format(useOffset=False) # turn off scientific plotting
    fig.suptitle(save_name, fontsize = 50)

    ax.scatter(x, y, color='green', s=20)

    ax.set_facecolor('white')
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.xlabel('Size of Order', fontsize=40)
    plt.ylabel('% Reordered', fontsize=40)
    ax.autoscale(enable=True, axis='both', tight=None)
    
    plt.subplots_adjust(bottom = 0.2, top = 0.8)
    fig.savefig(path+save_name +'.png')
    plt.close()
    
    return

'''                       #######################
#----------------------   ## ---    MAIN   --- ##     --------------------------
                          #######################
'''
if __name__ == '__main__':  
    path = '/Users/judyjinn/Python/CDIPS/instacart'
    
    prior_concat, train_concat, orders_train, orders_test = open_concat_sets(path)
    priors_in_train, orders_train_info = open_training_sets(path)
    PriorsInTrain_withIDs = pd.read_csv(path+'/training/PriorsInTrain_withIDs.csv')

    
    k_fold = pd.read_csv(path+'/k-fold.csv')
    save_name = '5-fold Cross Validation Forest Size'
    graph_scatter(path, k_fold['Forest Size'], k_fold['F1 Score'], save_name)
    
    test =  PriorsInTrain_withIDs.groupby('user_id')[
        'avg_order_size', 'pd_overall_reorder_rate'].mean().reset_index()
    save_name = 'Rate of Reordering & Order Size' 
    graph_scatter(
        path, test['avg_order_size'], test['pd_overall_reorder_rate'], save_name)
    
    avg_order_size = PriorsInTrain_withIDs.groupby(
        'user_id')['avg_order_size'].mean().reset_index()
    save_name = 'Average Order Size'
    bins = 50    
    graph_histogram(path, avg_order_size['avg_order_size'], save_name, bins)
    
    
    avg_size_of_order_with_prod = PriorsInTrain_withIDs.groupby(
        'user_id')['avg_size_of_order_with_prod'].mean().reset_index()
    save_name = 'Average Order Size \n Containing Reorders'
    bins = 50    
    graph_histogram(path, avg_size_of_order_with_prod[
        'avg_size_of_order_with_prod'], save_name, bins)
    
    tot_uniq_prod = PriorsInTrain_withIDs.groupby(
        'user_id')['tot_uniq_prod'].mean().reset_index()
    save_name = 'User Unique Products'
    bins = 50    
    graph_histogram(path, tot_uniq_prod['tot_uniq_prod'], save_name, bins)
    
    dow = orders_train['order_dow'].value_counts().reset_index()
    dow.columns=['Day', 'count']
    x_labels = ['Sat','Sun','Mon','Tues','Wed','Thurs','Fri']
    save_name = 'Day of Order Purchase'
    graph_bar(path, dow['Day'], dow['count'], x_labels, save_name, bins)
    
    hour = orders_train['order_hour_of_day'].value_counts().reset_index()
    x_labels = hour['index']
    save_name = 'Time of Order Purchase'
    graph_bar(path, hour['index'], hour['order_hour_of_day'],
         x_labels, save_name, bins
         )
    
    avg_days_bwn_orders = PriorsInTrain_withIDs.groupby(
        'user_id')['avg_days_bwn_orders'].mean().reset_index()
    save_name = 'Days Between Orders'
    bins = 50    
    graph_histogram(path, avg_days_bwn_orders['avg_days_bwn_orders'], 
        save_name, bins
        )
    
    
    
    

