'''                         #######################
#------------------------   ## ---   Set up  --- ##     ------------------------
                            #######################
'''

import os 
import re #used for natural sort function
import pandas as pd  #data frames
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import tkinter as Tk #opening files
from tkinter import filedialog
import numpy as np
import scipy.stats as stats
import scipy.io as sio
import sklearn
import math

 # auto-detect width of terminal for displaying dataframes
pd.set_option('display.max_columns',0)
plt.ion() # Solves that stupid iPython matplotlib hang problem
pd.options.mode.chained_assignment = None 
np.set_printoptions(suppress=True)



'''                          #######################
#-------------------------   ## --- Functions --- ##     -----------------------
                             #######################
'''

def open_CSVs(path):
    path = path+'/instacart_2017_05_01'
    # Open CSVs
    # This takes a while, avoid running this cell.
    print('loading aisles')
    aisles_df = pd.read_csv(path+"/aisles.csv", dtype= { 
            'aisle_id': np.int32, \
            'aisle': 'category'})
    print('loading departments')
    departments_df = pd.read_csv(path+"/departments.csv", dtype= {   
            'department_id': np.int32, \
            'department': 'category'})
    print('loading order products prior')
    order_products_prior_df = pd.read_csv(path+"/order_products__prior.csv", dtype={ 
            'order_id': np.uint32, \
            'product_id': np.uint32, \
            'add_to_cart_order': np.uint8, \
            'reordered': np.uint8})
    print('loading order products train')
    order_products_train_df = pd.read_csv(path+"/order_products__train.csv", dtype={ 
            'order_id': np.uint32, \
            'product_id': np.uint32, \
            'add_to_cart_order': np.uint8, \
            'reordered': np.uint8})
    print('loading orders')
    orders_df = pd.read_csv(path+"/orders.csv", dtype={ 
            'order_id': np.uint32, \
            'user_id': np.uint32, \
            'eval_set': 'category', \
            'order_number': np.uint8, \
            'order_dow': np.uint8, \
            'order_hour_of_day': np.uint8, \
            'days_since_prior_order': np.float32})
    print('loading products')
    products_df = pd.read_csv(path+"/products.csv", dtype={ 
            'product_id': np.uint32, \
            'product_name': 'category', \
            'aisle_id': np.int32, \
            'department_id': np.int32})
    sample_submission_df = pd.read_csv(path+"/sample_submission.csv")
    print('finished loading')
    
    return aisles_df, departments_df, order_products_prior_df, order_products_train_df, orders_df, products_df, sample_submission_df
    
def concat_CSVs(path, aisles_df, departments_df, order_products_prior_df, order_products_train_df, orders_df, products_df, sample_submission_df):
    path = path+'/instacart_2017_05_01'
    
    # Merge the product ID data frames so we know more about each product
    print('merging product information')
    product_info_df = pd.merge(products_df, departments_df, on='department_id')
    product_info_df = pd.merge(product_info_df, aisles_df, on='aisle_id')

    # Seperate out the prior orders, training set, and the test set
    print('merging order information')
    orders_prior = orders_df[orders_df['eval_set']=='prior']
    orders_train = orders_df[orders_df['eval_set']=='train']
    orders_test = orders_df[orders_df['eval_set']=='test']
    
    print('splitting orders to priors and training')
    orders_prior_concat = pd.merge(orders_prior, order_products_prior_df, on='order_id')
    orders_train_concat = pd.merge(orders_train, order_products_train_df, on='order_id')
    
    # Create full training set and test set for models
    print('concatinating full prior set')
    prior_concat = pd.merge(orders_prior_concat, product_info_df, on='product_id')
    print('concatinating full training set')
    train_concat =  pd.merge(orders_train_concat, product_info_df, on='product_id')
    
    print('saving files')
    prior_concat.to_csv(path+'/prior_concat.csv', sep=',', index=False)
    train_concat.to_csv(path+'/train_concat.csv', sep=',', index=False)
    orders_train.to_csv(path+'/orders_train.csv', sep=',', index=False)
    orders_test.to_csv(path+'/orders_test.csv', sep=',', index=False)
    
    return prior_concat, orders_train_concat, orders_train, orders_test

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
    
def subset_PandT(path, prior_concat, train_concat, user_cutoff):
    path = path+'/training/subset'
    
    # ---- Subset of  of priors set for exploration/setting up features--------
    # take a subset of the data so it's not a hellish data set first
    # For initial validation purposes, use only subset of users
    # that appear in both the priors and the training (which are 131209 of them)

    prior_concat['user_TF'] = prior_concat['user_id'].isin(train_concat['user_id']) 
    priors_intrain = prior_concat[prior_concat['user_TF']==True] 
    priors_test = prior_concat[prior_concat['user_TF']==False]
    
    # for sorting purposes , change user ID to int.
    copy_PIT = priors_intrain
    copy_PIT['user_id'] = copy_PIT['user_id'].astype(int)
    sub_Pintrain = copy_PIT[copy_PIT['user_id']<user_cutoff]
    sub_Pintrain.drop(['department','aisle', 'user_TF'], 
        axis=1,inplace=True)
        
    priors_test.drop(['department','aisle','eval_set', 
        'user_TF'], axis=1,inplace=True)
    
    copy_train = train_concat
    copy_train['user_id'] = copy_train['user_id'].astype(int)
    sub_train = copy_train[copy_train['user_id']<user_cutoff]
    sub_train.drop(['department','aisle'], 
        axis=1,inplace=True)
    
    ## Sanity check all users in Sub_Pintrain are in sub_train
    print('Sanity check, all users in prior subset are in the training subset')
    print(sub_Pintrain['user_id'].isin(sub_train['user_id']).value_counts())
    print('subset size of Pintrain and train_concat is',  len(sub_Pintrain), len(sub_train))
    print('saving both subsets and subset of users found only in priors, not in training')
    
    if (user_cutoff != 206210): 
        sub_Pintrain.to_csv(path+'/'+str(user_cutoff)+'_sub_Pintrain.csv', 
            sep=',', index=False)
        sub_train.to_csv(path+'/'+str(user_cutoff)+'_sub_train.csv', 
            sep=',', index=False)
    elif (user_cutoff == 206210):
        sub_Pintrain.to_csv(path+'/priors_in_train.csv', 
            sep=',', index=False)
        sub_train.to_csv(path+'/orders_train_full.csv', 
            sep=',', index=False)
        priors_test.to_csv(path+'/priors_in_test.csv', 
            sep=',', index=False)
        
    return
      
def open_subsets(path, user_cutoff):
    path = path+'/training/subset'
    print('opening sub_Pintrain')
    sub_Pintrain = pd.read_csv(path+'/'+str(user_cutoff)+'_sub_Pintrain.csv', 
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
    print('opening sub_train')
    sub_train = pd.read_csv(path+'/'+str(user_cutoff)+'_sub_train.csv', 
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
    return sub_Pintrain, sub_train 

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
    
def open_test_sets(path):
    path = path+'/training/subset'
     
    print('opening priors_in_test')
    priors_in_test = pd.read_csv(path+'/priors_in_test.csv',
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
    
    path = '/Users/judyjinn/Python/CDIPS/instacart/instacart_2017_05_01'
    print('opening orders_test')
    orders_test = pd.read_csv(path+'/orders_test.csv',
        dtype={
            'order_id': np.uint32,
            'user_id': np.uint32,
            'order_number': np.uint8,
            'order_dow': np.uint8,
            'order_hour_of_day': np.uint8,
            'days_since_prior_order': np.float32
        })
    print('finished opening')
    return priors_in_test, orders_test

def modes(df, key_cols, value_col, count_col):
    '''                                                                                                                                                                                                                                                                                                                                                              
    Pandas does not provide a `mode` aggregation function                                                                                                                                                                                                                                                                                                            
    for its `GroupBy` objects. This function is meant to fill                                                                                                                                                                                                                                                                                                        
    that gap, though the semantics are not exactly the same.                                                                                                                                                                                                                                                                                                         

    The input is a DataFrame with the columns `key_cols`                                                                                                                                                                                                                                                                                                             
    that you would like to group on, and the column                                                                                                                                                                                                                                                                                                                  
    `value_col` for which you would like to obtain the modes.                                                                                                                                                                                                                                                                                                        

    The output is a DataFrame with a record per group that has at least                                                                                                                                                                                                                                                                                              
    one mode (null values are not counted). The `key_cols` are included as                                                                                                                                                                                                                                                                                           
    columns, `value_col` contains lists indicating the modes for each group,                                                                                                                                                                                                                                                                                         
    and `count_col` indicates how many times each mode appeared in its group.                                                                                                                                                                                                                                                                                        
    '''
    return df.groupby(key_cols + [value_col]).size() \
             .to_frame(count_col).reset_index() \
             .groupby(key_cols + [count_col])[value_col].unique() \
             .to_frame().reset_index() \
             .sort_values(count_col, ascending=False) \
             .drop_duplicates(subset=key_cols)
             
def create_training_set(path, priors_concat, final_orders, user_cutoff):
    # ----------------------- Create design set ------------------------------
    '''
    prior_concat: The fully concatenated information about prior orders
    final_orders: fully concatenated information with just final orders
        AKA either eval_set labeled 'train' or 'test' 
    
    
    user_prod: Main DF storing most of the features.
    user_prod_ct: number of times each product was ordered by that user
    tot_uniq_prod: total number of unique products ordered by user
    tot_prod: total number of products ordered by user
    order_prod_ct: products in each order and number of those products
    order_size: size of each order
    avg_order_size: average size of a user's orders
    
    user_id: 
    product_id: 
    prod_ct: number of times each product was ordered by that user
    tot_uniq_prod: total number of unique products ordered by user
    tot_prod: otal number of products ordered by user
    avg_order_size: average size of a user's orders
    ** user_reorder_rate: average percent of an order which contains reordered products
    tot_num_orders: 
    prod_x_reordered: 
    prod_reorder_rate: 
    aisle_id: 
    department_id: 
    prod_days_since_reorder: 
    '''
    
    # dumb merging method, you can write it all in fewer lines later
    
    # --------- Feature Set for Training Order Size Classifier -----------------

    print('Calculating order_size')
    order_size = priors_concat.groupby(
        ['order_id', 'user_id']
        )['product_id'].nunique().reset_index(name='order_size')
    order_per_reordered = priors_concat.groupby(
        ['order_id', 'user_id','order_number']
        )['reordered'].mean().reset_index(
            name='order_per_reordered'
            ).sort_values('user_id')
    user_order = pd.merge(order_size, order_per_reordered,
        on=['user_id','order_id']
        )
    
    print('Calculating avg_per_reorders_in_order')
    avg_per_reorders_in_order = order_per_reordered.groupby(
        ['user_id']
        )['order_per_reordered'].mean().reset_index(name='avg_per_reorders_in_order').sort_values('user_id')
    user_order = pd.merge(user_order, avg_per_reorders_in_order,
        on=['user_id']
        )

    user_order = pd.merge(user_order, priors_concat[
            ['user_id', 'order_id','order_number',\
            'days_since_prior_order', 'order_dow', 'order_hour_of_day']
            ], on=['user_id', 'order_id', 'order_number']
        )
    user_order = user_order.groupby(
        ['order_id']
        ).mean().sort_values(
            by=['user_id','order_number']
            ).reset_index()

    print('Calculating num_reordered_items')
    num_reordered_items = priors_concat.groupby(['order_id', 'user_id'])['reordered'].sum().reset_index(name='num_reordered_items')
    user_order = pd.merge(user_order, num_reordered_items,
        on=['user_id','order_id']
        )

    print('Calculating avg_days_bwn_orders')
    avg_days_bwn_orders = priors_concat.groupby(
        ['order_id', 'user_id']
        )['days_since_prior_order'].mean().reset_index(
            name='avg_days_bwn_orders'
            )
    user_order = pd.merge(user_order, avg_days_bwn_orders,
        on=['user_id','order_id']
        )
            
            
    user_order = user_order.sort_values(by=['user_id','order_number'])
    
    
    print('Calculating avg_reordered_per_order')
    avg_reordered_per_order = user_order.groupby('user_id')['num_reordered_items'].mean().reset_index(name='avg_reordered_per_order')
    user_order = pd.merge(user_order, avg_reordered_per_order,
            on=['user_id']
            )
        
            
    

    print('Calculating order_dow_mode')
    # Find most common day of the week for the order
    # Be careful with this. If classifier gets wonky, take this part out first
    # Returns a tuple of the most frequent days
    order_dow_mode = modes(priors_concat, ['user_id'], 'order_dow', 'count')
    order_dow_mode = order_dow_mode.rename(index=str, columns={'order_dow': 'order_dow_mode'})
    # Tuple is expanded into X feature columns with days of week or NaN=999.0
    # 999.0 may end up caushing problems
    tmp_order_mode = order_dow_mode['order_dow_mode'].apply(pd.Series)
    tmp_order_mode = tmp_order_mode.fillna(value=999)
    tmp_order_mode = tmp_order_mode.astype(int)
    high_mode_num =  tmp_order_mode.shape[1]
    column_names = list(order_dow_mode.columns)
    for i in range(0,high_mode_num):
        column_names.append('order_dow_mode'+str(i))
    order_dow_mode = pd.concat([order_dow_mode,tmp_order_mode], axis=1)
    order_dow_mode.columns = column_names
    order_dow_mode = order_dow_mode.drop(['order_dow_mode','count'],1)
    # only use top 2 dow
    user_order = pd.merge(user_order, order_dow_mode.iloc[:,:4], 
            on=['user_id']
            )
            
        



    
    # ------ Feature Set for Training Binary Product Reorder Classifier --------

    print('Calculating tot_num_orders')
    tot_num_orders = priors_concat.groupby(
        ['user_id']
        )['order_number'].max().reset_index(name='tot_num_orders')
    prod_x_reordered = priors_concat.groupby(
        ['user_id', 'product_id']
        )['reordered'].sum().reset_index(name ='prod_x_reordered')    
    user_prod = pd.merge(
        tot_num_orders, prod_x_reordered, on=['user_id']
        )
    user_prod['prod_reorder_rate'] = \
        user_prod['prod_x_reordered']/(user_prod['tot_num_orders']-1)
    
    print('Calculating avg_size_of_order_with_prod')
    tmp =  priors_concat[['user_id', 'product_id','order_id']].sort_values(by='user_id')
    order_size = priors_concat.groupby(
        ['order_id', 'user_id']
        )['product_id'].nunique().reset_index(name='order_size')
    tmp_2 = pd.merge(tmp, order_size, on=['user_id','order_id'])
    avg_size_of_order_with_prod = tmp_2.groupby(
        ['user_id', 'product_id']
        )['order_size'].mean().reset_index(name='avg_size_of_order_with_prod')
    
    user_prod = pd.merge(user_prod, avg_size_of_order_with_prod, 
        on=['user_id', 'product_id']
        )




    # print('Calculating prod_dow_mode')
    # # Find most common day of the week for the product
    # # Be careful with this. If classifier gets wonky, take this part out first
    # # Returns a tuple of the most frequent days
    # prod_dow_mode = modes(priors_concat, ['user_id','product_id'], 'order_dow', 'count')
    # prod_dow_mode = prod_dow_mode.rename(index=str, columns={'order_dow': 'prod_dow_mode'})
    # # Tuple is expanded into X feature columns with days of week or NaN=999.0
    # # 999.0 may end up caushing problems
    # tmp_prod_mode = prod_dow_mode['prod_dow_mode'].apply(pd.Series, dtype=np.uint8)
    # tmp_prod_mode = tmp_prod_mode.fillna(value=999)
    # tmp_prod_mode = tmp_prod_mode.astype(int)
    # high_mode_num =  tmp_prod_mode.shape[1]
    # column_names = list(prod_dow_mode.columns)
    # for i in range(0,high_mode_num):
    #     column_names.append('prod_dow_mode'+str(i))
    # prod_dow_mode = pd.concat([prod_dow_mode,tmp_prod_mode], axis=1)
    # prod_dow_mode.columns = column_names
    # prod_dow_mode = prod_dow_mode.drop(['prod_dow_mode','count'],1)
    # user_prod = pd.merge(user_prod, prod_dow_mode.iloc[:,:4],
    #         on=['user_id', 'product_id']
    #         )
    #
            
    # print('Calculating average days between last product purchase...')
    # # goals:
    # #   1) from validation set find days passed since product was last ordered
    # #   2) find the average number of days passed between reordering that product
    #
    # # Take only the value of the max order number
    # cumm_days_since_last_order = priors_concat[
    #     ['user_id', 'order_number', 'days_since_prior_order']
    #     ].sort_values(by=['user_id', 'order_number'])
    # cumm_days_since_last_order = cumm_days_since_last_order.groupby(
    #     ['user_id', 'order_number', 'days_since_prior_order']
    #     )['order_number'].nunique().reset_index(name='count')
    # cumm_days_since_last_order = cumm_days_since_last_order.sort_values(
    #     by=['user_id', 'order_number']
    #     )
    #
    # # Find days since last order of that product
    # days_passed_btwn_prod = priors_concat[
    #     ['user_id', 'product_id','order_number','days_since_prior_order']
    #     ].sort_values(by=['user_id', 'product_id'])
    # # Any NaN in order_num_diff values mean it is the first purchase for that product
    # days_passed_btwn_prod['order_num_diff'] = days_passed_btwn_prod.groupby(
    #     ['user_id', 'product_id']
    #     )['order_number'].diff()

    # days_passed = []
    # for i in days_passed_btwn_prod.index:
    #     if (math.isnan(days_passed_btwn_prod['order_num_diff'][i]) == True):
    #         days_passed.append(math.nan)
    #     elif (days_passed_btwn_prod['order_num_diff'][i] == 1.0):
    #         days_passed.append(days_passed_btwn_prod['days_since_prior_order'][i])
    #     elif (days_passed_btwn_prod['order_num_diff'][i] > 1.0):
    #         user_id = days_passed_btwn_prod['user_id'][i]
    #         prior_order_num = days_passed_btwn_prod['order_number'][i-1]
    #         curr_order_num = days_passed_btwn_prod['order_number'][i]
    #
    #         cumm_user = cumm_days_since_last_order[cumm_days_since_last_order['user_id']==user_id]
    #         if(prior_order_num==1):
    #             prior_order_num_index = cumm_user['order_number'][
    #                 (cumm_user['order_number']==2)
    #                 ].index[0]
    #             curr_order_num_index = cumm_user['order_number'][
    #                 (cumm_user['order_number']==curr_order_num)
    #                 ].index[0]
    #             days_passed.append(
    #                 cumm_user.loc[
    #                     prior_order_num_index: curr_order_num_index
    #                     ]['days_since_prior_order'].sum()
    #                 )
    #         else:
    #             prior_order_num_index = cumm_user['order_number'][
    #                 (cumm_user['order_number']==prior_order_num)
    #                 ].index[0]
    #             curr_order_num_index = cumm_user['order_number'][
    #                 (cumm_user['order_number']==curr_order_num)
    #                 ].index[0]
    #             days_passed.append(
    #                 cumm_user.loc[
    #                     prior_order_num_index+1 : curr_order_num_index
    #                     ]['days_since_prior_order'].sum()
    #             )
    #
    # days_passed_btwn_prod['days_since_last_purchase'] = days_passed
    # avg_days_passed_btwn_prod = days_passed_btwn_prod.groupby(
    #     ['user_id', 'product_id']
    #     )['days_since_last_purchase'].mean().reset_index(
    #         name='avg_days_btwn_prod'
    #         )
    # user_prod = pd.merge(user_prod, avg_days_passed_btwn_prod,
    #         on=['user_id', 'product_id']
    #         )
    # user_prod['avg_days_btwn_prod'] = user_prod['avg_days_btwn_prod'].fillna(value=999.0)
    #
    #
    # print('Done calculating average days passed')
    #
    
    print('Calculating average days between last order')
    
    
    print('Calculating user_prod_ct')
    user_prod_ct = priors_concat.groupby(
        ['user_id']
        )['product_id'].value_counts().reset_index(name='prod_ct')
    user_prod = pd.merge(user_prod, user_prod_ct, on=['user_id', 'product_id'])
    
    print('Calculating tot_uniq_prod')
    tot_uniq_prod = priors_concat.groupby(
        ['user_id']
        )['product_id'].nunique().reset_index(name='tot_uniq_prod')
    user_prod = pd.merge(user_prod, tot_uniq_prod, on='user_id')
    
    print('Calculating tot_prod')
    tot_prod = user_prod_ct.groupby(
        ['user_id']
        )['prod_ct'].sum().reset_index(name='tot_prod')
    user_prod = pd.merge(user_prod, tot_prod, on='user_id')

    print('Calculating order_prod_ct')
    order_prod_ct = priors_concat.groupby(
        ['order_id', 'user_id']
        )['product_id'].value_counts().reset_index(name='order_prod_ct')
    order_size = order_prod_ct.groupby(['order_id', 'user_id']
        )['order_prod_ct'].sum().reset_index(name='order_size')
    avg_order_size = order_size.groupby(
        ['user_id']
        )['order_size'].mean().reset_index(name='avg_order_size')
    user_prod = pd.merge(user_prod, avg_order_size, on = 'user_id')
    
    user_prod['pd_overall_reorder_rate'] = user_prod['prod_ct']/user_prod['tot_prod']

    
    # concat some columns from the user_orders data frame
    cols = ['user_id', 'avg_per_reorders_in_order', 'avg_days_bwn_orders',
        'order_dow_mode0', 'order_dow_mode1']
    tmp = user_order[cols]
    tmp = tmp.groupby('user_id').mean().reset_index()
        
    user_prod = pd.merge(user_prod, tmp, on = 'user_id')
    

    
    # Apply features relating the user's last order to be predicted
    
    tmp = final_orders[['user_id', 'order_dow',
           'order_hour_of_day', 'days_since_prior_order']]
    tmp = tmp.groupby('user_id').mean().reset_index()
    user_prod = pd.merge(user_prod, tmp, on=['user_id']).sort_values(by=['user_id'])
    
    # Always sort by user_id and product_id before final steps.
    final_orders = final_orders.sort_values(by=['user_id','product_id'])
    user_prod = user_prod.sort_values(by=['user_id','product_id'])
    
    
    # Rearrange the array in whatever way you like
    cols = user_prod.columns.tolist()
    cols.insert(1, cols.pop(cols.index('product_id')))
    
    user_prod = user_prod.reindex(columns= cols)
    
    # Create labels set
    final_orders= final_orders.sort_values(by=['user_id','product_id'])
    tmp_final_orders = final_orders[['user_id','product_id', 'reordered']]
    user_prod = pd.merge(user_prod, tmp_final_orders, how='left', on=['user_id','product_id'])
    user_prod['reordered'] = user_prod['reordered'].fillna(value=0.0)
    
    # Save file
    print('Saving user_prod')
    # user_prod.to_csv(path+'/training/'+str(user_cutoff)+'_user_prod_withIDs.csv',
    #         sep=',', index=False)

    print('user_cutoff is', user_cutoff)
    if (user_cutoff == 'priors_in_train'):
        user_prod.to_csv(path+'/training/PriorsInTrain__withIDs.csv', 
                sep=',', index=False)
    elif (user_cutoff == 'PriorTest_priors'):
        user_prod.to_csv(path+'/training/PriorsInTest_4train__withIDs.csv', 
                sep=',', index=False)
    elif (user_cutoff == 'orders_test'):
        user_prod.to_csv(path+'/training/PriorsInTest_4testing__withIDs.csv', 
                sep=',', index=False)
    else: 
        user_prod.to_csv(path+'/training/PriorsInTrain'+str(user_cutoff)+'_withIDs.csv', 
                sep=',', index=False)
    
    print('Finished creating training set!')
                
    
    return
    


def create_testing_set(path, all_priors, final_orders, user_cutoff):

    print('Calculating order_size')
    order_size = all_priors.groupby(
        ['order_id', 'user_id']
        )['product_id'].nunique().reset_index(name='order_size')
    order_per_reordered = all_priors.groupby(
        ['order_id', 'user_id','order_number']
        )['reordered'].mean().reset_index(
            name='order_per_reordered'
            ).sort_values('user_id')
    user_order = pd.merge(order_size, order_per_reordered,
        on=['user_id','order_id']
        )
    
    print('Calculating avg_per_reorders_in_order')
    avg_per_reorders_in_order = order_per_reordered.groupby(
        ['user_id']
        )['order_per_reordered'].mean().reset_index(name='avg_per_reorders_in_order').sort_values('user_id')
    user_order = pd.merge(user_order, avg_per_reorders_in_order,
        on=['user_id']
        )

    user_order = pd.merge(user_order, all_priors[
            ['user_id', 'order_id','order_number',\
            'days_since_prior_order', 'order_dow', 'order_hour_of_day']
            ], on=['user_id', 'order_id', 'order_number']
        )
    user_order = user_order.groupby(
        ['order_id']
        ).mean().sort_values(
            by=['user_id','order_number']
            ).reset_index()

    print('Calculating num_reordered_items')
    num_reordered_items = all_priors.groupby(['order_id', 'user_id'])['reordered'].sum().reset_index(name='num_reordered_items')
    user_order = pd.merge(user_order, num_reordered_items,
        on=['user_id','order_id']
        )

    print('Calculating avg_days_bwn_orders')
    avg_days_bwn_orders = all_priors.groupby(
        ['order_id', 'user_id']
        )['days_since_prior_order'].mean().reset_index(
            name='avg_days_bwn_orders'
            )
    user_order = pd.merge(user_order, avg_days_bwn_orders,
        on=['user_id','order_id']
        )
    
    user_order = user_order.sort_values(by=['user_id','order_number'])
    
    
    print('Calculating avg_reordered_per_order')
    avg_reordered_per_order = user_order.groupby('user_id')['num_reordered_items'].mean().reset_index(name='avg_reordered_per_order')
    user_order = pd.merge(user_order, avg_reordered_per_order,
            on=['user_id']
            )

    print('Calculating order_dow_mode')
    # Find most common day of the week for the order
    # Be careful with this. If classifier gets wonky, take this part out first
    # Returns a tuple of the most frequent days
    order_dow_mode = modes(all_priors, ['user_id'], 'order_dow', 'count')
    order_dow_mode = order_dow_mode.rename(index=str, columns={'order_dow': 'order_dow_mode'})
    # Tuple is expanded into X feature columns with days of week or NaN=999.0
    # 999.0 may end up caushing problems
    tmp_order_mode = order_dow_mode['order_dow_mode'].apply(pd.Series)
    tmp_order_mode = tmp_order_mode.fillna(value=999)
    tmp_order_mode = tmp_order_mode.astype(int)
    high_mode_num =  tmp_order_mode.shape[1]
    column_names = list(order_dow_mode.columns)
    for i in range(0,high_mode_num):
        column_names.append('order_dow_mode'+str(i))
    order_dow_mode = pd.concat([order_dow_mode,tmp_order_mode], axis=1)
    order_dow_mode.columns = column_names
    order_dow_mode = order_dow_mode.drop(['order_dow_mode','count'],1)
    # only use top 2 dow
    user_order = pd.merge(user_order, order_dow_mode.iloc[:,:4], 
            on=['user_id']
            )


    # ------ Feature Set for  Binary Product Reorder Classifier --------

    print('Calculating tot_num_orders')
    tot_num_orders = all_priors.groupby(
        ['user_id']
        )['order_number'].max().reset_index(name='tot_num_orders')
    prod_x_reordered = all_priors.groupby(
        ['user_id', 'product_id']
        )['reordered'].sum().reset_index(name ='prod_x_reordered')    
    user_prod = pd.merge(
        tot_num_orders, prod_x_reordered, on=['user_id']
        )
    user_prod['prod_reorder_rate'] = \
        user_prod['prod_x_reordered']/(user_prod['tot_num_orders']-1)
    
    print('Calculating avg_size_of_order_with_prod')
    tmp =  all_priors[['user_id', 'product_id','order_id']].sort_values(by='user_id')
    order_size = all_priors.groupby(
        ['order_id', 'user_id']
        )['product_id'].nunique().reset_index(name='order_size')
    tmp_2 = pd.merge(tmp, order_size, on=['user_id','order_id'])
    avg_size_of_order_with_prod = tmp_2.groupby(
        ['user_id', 'product_id']
        )['order_size'].mean().reset_index(name='avg_size_of_order_with_prod')
    
    user_prod = pd.merge(user_prod, avg_size_of_order_with_prod, 
        on=['user_id', 'product_id']
        )


    #
    # print('Calculating prod_dow_mode')
    # # Find most common day of the week for the product
    # # Be careful with this. If classifier gets wonky, take this part out first
    # # Returns a tuple of the most frequent days
    # prod_dow_mode = modes(all_priors, ['user_id','product_id'], 'order_dow', 'count')
    # prod_dow_mode = prod_dow_mode.rename(index=str, columns={'order_dow': 'prod_dow_mode'})
    # # Tuple is expanded into X feature columns with days of week or NaN=999.0
    # # 999.0 may end up caushing problems
    # tmp_prod_mode = prod_dow_mode['prod_dow_mode'].apply(pd.Series, dtype=np.uint8)
    # tmp_prod_mode = tmp_prod_mode.fillna(value=999)
    # tmp_prod_mode = tmp_prod_mode.astype(int)
    # high_mode_num =  tmp_prod_mode.shape[1]
    # column_names = list(prod_dow_mode.columns)
    # for i in range(0,high_mode_num):
    #     column_names.append('prod_dow_mode'+str(i))
    # prod_dow_mode = pd.concat([prod_dow_mode,tmp_prod_mode], axis=1)
    # prod_dow_mode.columns = column_names
    # prod_dow_mode = prod_dow_mode.drop(['prod_dow_mode','count'],1)
    # user_prod = pd.merge(user_prod, prod_dow_mode.iloc[:,:4],
    #         on=['user_id', 'product_id']
    #         )
    #
            
    # print('Calculating average days between last product purchase...')
    # # goals:
    # #   1) from validation set find days passed since product was last ordered
    # #   2) find the average number of days passed between reordering that product
    #
    # # Take only the value of the max order number
    # cumm_days_since_last_order = all_priors[
    #     ['user_id', 'order_number', 'days_since_prior_order']
    #     ].sort_values(by=['user_id', 'order_number'])
    # cumm_days_since_last_order = cumm_days_since_last_order.groupby(
    #     ['user_id', 'order_number', 'days_since_prior_order']
    #     )['order_number'].nunique().reset_index(name='count')
    # cumm_days_since_last_order = cumm_days_since_last_order.sort_values(
    #     by=['user_id', 'order_number']
    #     )
    #
    # # Find days since last order of that product
    # days_passed_btwn_prod = all_priors[
    #     ['user_id', 'product_id','order_number','days_since_prior_order']
    #     ].sort_values(by=['user_id', 'product_id'])
    # # Any NaN in order_num_diff values mean it is the first purchase for that product
    # days_passed_btwn_prod['order_num_diff'] = days_passed_btwn_prod.groupby(
    #     ['user_id', 'product_id']
    #     )['order_number'].diff()
    #
    # days_passed = []
    # for i in days_passed_btwn_prod.index:
    #     if (math.isnan(days_passed_btwn_prod['order_num_diff'][i]) == True):
    #         days_passed.append(math.nan)
    #     elif (days_passed_btwn_prod['order_num_diff'][i] == 1.0):
    #         days_passed.append(days_passed_btwn_prod['days_since_prior_order'][i])
    #     elif (days_passed_btwn_prod['order_num_diff'][i] > 1.0):
    #         user_id = days_passed_btwn_prod['user_id'][i]
    #         prior_order_num = days_passed_btwn_prod['order_number'][i-1]
    #         curr_order_num = days_passed_btwn_prod['order_number'][i]
    #
    #         cumm_user = cumm_days_since_last_order[cumm_days_since_last_order['user_id']==user_id]
    #         if(prior_order_num==1):
    #             prior_order_num_index = cumm_user['order_number'][
    #                 (cumm_user['order_number']==2)
    #                 ].index[0]
    #             curr_order_num_index = cumm_user['order_number'][
    #                 (cumm_user['order_number']==curr_order_num)
    #                 ].index[0]
    #             days_passed.append(
    #                 cumm_user.loc[
    #                     prior_order_num_index: curr_order_num_index
    #                     ]['days_since_prior_order'].sum()
    #                 )
    #         else:
    #             prior_order_num_index = cumm_user['order_number'][
    #                 (cumm_user['order_number']==prior_order_num)
    #                 ].index[0]
    #             curr_order_num_index = cumm_user['order_number'][
    #                 (cumm_user['order_number']==curr_order_num)
    #                 ].index[0]
    #             days_passed.append(
    #                 cumm_user.loc[
    #                     prior_order_num_index+1 : curr_order_num_index
    #                     ]['days_since_prior_order'].sum()
    #             )
    #
    # days_passed_btwn_prod['days_since_last_purchase'] = days_passed
    # avg_days_passed_btwn_prod = days_passed_btwn_prod.groupby(
    #     ['user_id', 'product_id']
    #     )['days_since_last_purchase'].mean().reset_index(
    #         name='avg_days_btwn_prod'
    #         )
    # user_prod = pd.merge(user_prod, avg_days_passed_btwn_prod,
    #         on=['user_id', 'product_id']
    #         )
    # user_prod['avg_days_btwn_prod'] = user_prod['avg_days_btwn_prod'].fillna(value=999.0)
    #
    #
    # print('Done calculating average days passed')
    
    
    print('Calculating user_prod_ct')
    user_prod_ct = all_priors.groupby(
        ['user_id']
        )['product_id'].value_counts().reset_index(name='prod_ct')
    user_prod = pd.merge(user_prod, user_prod_ct, on=['user_id', 'product_id'])
    
    print('Calculating tot_uniq_prod')
    tot_uniq_prod = all_priors.groupby(
        ['user_id']
        )['product_id'].nunique().reset_index(name='tot_uniq_prod')
    user_prod = pd.merge(user_prod, tot_uniq_prod, on='user_id')
    
    print('Calculating tot_prod')
    tot_prod = user_prod_ct.groupby(
        ['user_id']
        )['prod_ct'].sum().reset_index(name='tot_prod')
    user_prod = pd.merge(user_prod, tot_prod, on='user_id')

    print('Calculating order_prod_ct')
    order_prod_ct = all_priors.groupby(
        ['order_id', 'user_id']
        )['product_id'].value_counts().reset_index(name='order_prod_ct')
    order_size = order_prod_ct.groupby(['order_id', 'user_id']
        )['order_prod_ct'].sum().reset_index(name='order_size')
    avg_order_size = order_size.groupby(
        ['user_id']
        )['order_size'].mean().reset_index(name='avg_order_size')
    user_prod = pd.merge(user_prod, avg_order_size, on = 'user_id')
    
    user_prod['pd_overall_reorder_rate'] = user_prod['prod_ct']/user_prod['tot_prod']

    
    # concat some columns from the user_orders data frame
    cols = ['user_id', 'avg_per_reorders_in_order', 'avg_days_bwn_orders',
        'order_dow_mode0', 'order_dow_mode1'] # + list(user_order.loc[:,'order_dow_mode0':])
    tmp = user_order[cols]
    tmp = tmp.groupby('user_id').mean().reset_index()
        
    user_prod = pd.merge(user_prod, tmp, on = 'user_id')
    
    # Apply features relating the user's last order to be predicted
    
    tmp = final_orders[['user_id', 'order_dow',
           'order_hour_of_day', 'days_since_prior_order']]
    tmp = tmp.groupby('user_id').mean().reset_index()
    user_prod = pd.merge(user_prod, tmp, on=['user_id']).sort_values(by=['user_id'])
    
    # Always sort by user_id and product_id before final steps.
    final_orders = final_orders.sort_values(by=['user_id'])
    user_prod = user_prod.sort_values(by=['user_id','product_id'])

    # Rearrange the array in whatever way you like
    cols = user_prod.columns.tolist()
    cols.insert(1, cols.pop(cols.index('product_id')))
    
    user_prod = user_prod.reindex(columns= cols)
    
    # Save file
    print('Saving user_prod')

    print('user_cutoff is', user_cutoff)
    ## This is the one to put into the classifer for submission
    if (user_cutoff == 'test_samples'):
        user_prod.to_csv(path+'/test/test_samples_withIDs.csv', 
                sep=',', index=False)
                
    print('Finished creating test set!')

    
    
    
    return


def split_prior_test(path, priors_in_test):
    path = path+'/training/'
    PriorTest_last_orders = priors_in_test
    PriorTest_priors = priors_in_test
    
    PriorTest_last_orders = priors_in_test
    PriorTest_priors = priors_in_test
    
    print('splitting priors in test into groups for training')
    # Split the priors_in_test so the last order can provide labels for training
    max_ord_num_df = priors_in_test.groupby('user_id')['order_number'].max().reset_index()
    
    for i in max_ord_num_df.index:
    # for i in [0,1]:
        user = max_ord_num_df['user_id'][i]
        max_ord_num = max_ord_num_df['order_number'][i]
        
        PriorTest_last_orders = \
            PriorTest_last_orders[PriorTest_last_orders['user_id']!=user]
        
        user_last_order = priors_in_test[\
                (priors_in_test['user_id']==user) &
                (priors_in_test['order_number']==max_ord_num)
                ]
        PriorTest_last_orders= pd.concat([PriorTest_last_orders,user_last_order], axis=0)

        PriorTest_priors = \
            PriorTest_priors[PriorTest_priors['user_id']!=user]
        
        user_priors = priors_in_test[ \
                (priors_in_test['user_id']==user) &
                (priors_in_test['order_number']!= max_ord_num)
                ]
                
        PriorTest_priors= pd.concat([PriorTest_priors, user_priors], axis=0)
        
    PriorTest_priors = PriorTest_priors.sort_values(['user_id', 'order_number'])
    PriorTest_last_orders = PriorTest_last_orders.sort_values(['user_id', 'order_number'])
    
    PriorTest_priors.to_csv(path+'/PriorsInTest_priors.csv', sep=',', index=False)
    PriorTest_last_orders.to_csv(path+'/PriorsInTest_last_orders.csv', sep=',', index=False)
    
    return PriorTest_priors, PriorTest_last_orders


def open_split_prior_test_sets(path):
    path = path+'/training/'
     
    print('opening PriorTest_priors')
    PriorsInTest_priors = pd.read_csv(path+'/PriorsInTest_priors.csv',
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
    
    print('opening PriorTest_last_orders')
    PriorsInTest_last_orders = pd.read_csv(path+'/PriorsInTest_last_orders.csv',
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
    return PriorsInTest_priors, PriorsInTest_last_orders

def main():
    
    path = '/Users/judyjinn/Python/CDIPS/instacart'
    
    ## ------- If Loading all Data from the beginning use this section ---------
    aisles_df, departments_df, order_products_prior_df, \
    order_products_train_df, orders_df, products_df, sample_submission_df = \
        open_CSVs(path)

    # This takes ages to concatenate. Don't run it again after the first time!
    prior_concat, orders_train_concat, orders_train, orders_test = concat_CSVs(
        path, aisles_df, departments_df, order_products_prior_df,
        order_products_train_df, orders_df, products_df, sample_submission_df
        )
    
    ## ----------------------- Load  data sets -----------------------------
    prior_concat, train_concat, orders_train, orders_test = open_concat_sets(path)
    

    # # --------------- Create Subsets for small testing -----------------------
    # choose the subset size. Options are users < 500 and 5000
    
    user_cutoff = 5000
    # print('Subsetting by user ID lower than', user_cutoff)
    # subset_PandT(path, prior_concat, train_concat, user_cutoff)

    sub_Pintrain, sub_train = open_subsets(path, user_cutoff)

    create_training_set(path, sub_Pintrain, sub_train, user_cutoff)


    
    
    # -------------------------- Full Training Set -----------------------------
    user_cutoff = 206210

    # Find all users that are located in train
    # All 'test' users are saved in 'priors_test'
    print('Subsetting by user ID lower than', user_cutoff, 'and found in train')
    subset_PandT(path, prior_concat, train_concat, user_cutoff)

    priors_in_train, orders_train_info = open_training_sets(path)

    # Create the full training data set for 131209 users
    # Begin by matching priors_train with orders train information
    print('Creating Training set with priors found in train and orders in train')
    create_training_set(path, priors_in_train, orders_train_info, 'priors_in_train')
    
    priors_in_test, orders_test = open_test_sets(path)

    # Split the priors_in_test so the last order can provide labels for training. It will save.. Then never run it again.
    split_prior_test(path, priors_in_test)
    
    # Open split training sets
    PriorTest_priors, PriorTest_last_orders = open_split_prior_test_sets(path)

    print('Creating training set with priors found in test and final orders created from priors in test')
    create_training_set(path, PriorTest_priors, PriorTest_last_orders, 'PriorTest_priors')

    # print('Concatenating training set of priors-train with priors-lastorder-test')
    # PriorsInTrain_user_prod_withIDs =
    #     pd.read_csv(path+'/training/PriorsInTrain_user_prod_withIDs.csv')


    
    # -------------------------- Full Test Set -----------------------------

    priors_in_test, orders_test = open_test_sets(path)

    print('Creating test set with final priors found in test and no labels')
    create_testing_set(path, priors_in_test, orders_test, 'test_samples')


    print('Concatenating training set of priors-train with priors-lastorder-test')

    return

'''                       #######################
#----------------------   ## ---    MAIN   --- ##     --------------------------
                          #######################
'''
    
# Note to self, ALL information needed for creating training sets are in:
# priors_in_train.csv, train.csv, priors_in_test

if __name__ == '__main__': 
    
    # main()
    
     # -------------------------- TEST ZONE -----------------------------
    path = '/Users/judyjinn/Python/CDIPS/instacart'
    
    
    priors_in_test, orders_test = open_test_sets(path)

    # Split the priors_in_test so the last order can provide labels for training. It will save.. Then never run it again.
    PriorTest_priors, PriorTest_last_orders = split_prior_test(path, priors_in_test)
    
    
    # Quick sanity checks
    # Making sure users match
    # np.array_equal(priors_in_test['user_id'].sort_values().unique() , orders_test['user_id'].sort_values().unique())    
    

    #
    # orders_test = pd.read_csv('/Users/judyjinn/Python/CDIPS/instacart/instacart_2017_05_01/orders_test.csv')
    # user_prod_withIDs_500 = pd.read_csv('/Users/judyjinn/Python/CDIPS/instacart/training/500_user_prod_withIDs.csv')
    #
    
    
    # # Testing split function for speed
    # priors_in_test, orders_test = open_test_sets(path)
    #
    # copy_PriorTest = priors_in_test
    # copy_PriorTest['user_id'] = copy_PriorTest['user_id'].astype(int)
    # sub_PriorTest_5000 = copy_PriorTest[copy_PriorTest['user_id']<5000]
    #
    # PriorTest_priors, PriorTest_last_orders = split_prior_test(path, sub_PriorTest_5000)
    #
    # max_ord_num_df = sub_PriorTest_5000.groupby('user_id')['order_number'].max().reset_index()
    # max_ord_num_df.tail(5)
    # PriorTest_priors.groupby(['user_id'])['order_number'].max().tail(5)
    # PriorTest_last_orders.groupby(['user_id'])['order_number'].max().tail(5)




