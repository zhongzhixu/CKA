import collections
import os
import numpy as np
import pandas as pd
np.random.seed(0)
sf_test = False

#%%
def load_data(args):
    train_data, eval_data, test_data, user_history_dict_ep1, user_history_dict_ep2, user_history_dict_ep3 = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    ripple_set = get_ripple_set(args, kg, user_history_dict_ep1, user_history_dict_ep2, user_history_dict_ep3)
    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    hrs_file = './DATA/data_sample.csv'
    hrs_df = pd.read_csv(hrs_file)
    return dataset_split(hrs_df)

def dataset_split(hrs_df):
    print('splitting dataset ...')
    #+++for general
    eval_ratio = 0.1
    test_ratio = 0.2
    #+++for self-harm
    sf_test_ratio = 0.99
    n_pat = len(set(hrs_df['PATID']))
    patid = list(set(hrs_df['PATID']))
    
    if sf_test:
        sf_patid = pd.read_csv('./DATA/sf_pat_newindex.csv', header=None)
        sf_patid = list(sf_patid.iloc[:,0])
        n_sf_pat = len(sf_patid)
        eval_indices= np.random.choice(sf_patid, size=int(n_sf_pat * sf_test_ratio), replace=False)
        left = set(hrs_df['PATID']) - set(eval_indices)
        test_indices =  eval_indices
        train_indices = list(left - set(test_indices)) 
        
    else:
        eval_indices = np.random.choice(patid, size=int(n_pat * eval_ratio), replace=False)
        left = set(hrs_df['PATID']) - set(eval_indices)    
        test_indices = np.random.choice(list(left), size=int(n_pat * test_ratio), replace=False)    
        train_indices = list(left - set(test_indices)) 
    
    # train_data, eval_data, test_data
    
    #train_test_df = hrs_df[hrs_df['episode']==-1]
    train_test_df = hrs_df
    
    train_df = train_test_df[train_test_df['PATID'].isin(train_indices)]
    train_data = train_df[['PATID', 'diag', 'label', 'episode']].values
    
    test_df = train_test_df[train_test_df['PATID'].isin(test_indices)]    
    test_data = test_df[['PATID', 'diag', 'label', 'episode']].values
    
    eval_df = train_test_df[train_test_df['PATID'].isin(eval_indices)]
    eval_data = eval_df[['PATID', 'diag', 'label', 'episode']].values
    
    #user_history_dict
    #train test evaluate all have user_history_dict
    
    user_history_dict_ep1,user_history_dict_ep2,user_history_dict_ep3 = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    
    hrs_df_ep1 = hrs_df[hrs_df['episode']==1]
    patid, diag = list(hrs_df_ep1['PATID']), list(hrs_df_ep1['diag'])
    for i, j in zip(patid, diag):
        user_history_dict_ep1[i].append(j)
        
    hrs_df_ep2 = hrs_df[hrs_df['episode']==2]
    patid, diag = list(hrs_df_ep2['PATID']), list(hrs_df_ep2['diag'])
    for i, j in zip(patid, diag):
        user_history_dict_ep2[i].append(j)
        
    hrs_df_ep3 = hrs_df[hrs_df['episode']==3]
    patid, diag = list(hrs_df_ep3['PATID']), list(hrs_df_ep3['diag'])
    for i, j in zip(patid, diag):
        user_history_dict_ep3[i].append(j)

    return train_data, eval_data, test_data, user_history_dict_ep1, user_history_dict_ep2, user_history_dict_ep3

def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = './DATA'  + '/kg_final' # from preprocess.py

    kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
    np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg

def get_ripple_set(args, kg, user_history_dict_ep1, user_history_dict_ep2, user_history_dict_ep3 ):
    print('constructing ripple set ...')
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    # ripple_set = collections.defaultdict(list)
    ripple_set_ep1, ripple_set_ep2, ripple_set_ep3 =  collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    
    # ripple_set_ep1
    for user in user_history_dict_ep1:
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict_ep1[user]
            else:
                tails_of_last_hop = ripple_set_ep1[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])
            
            if len(memories_h) == 0:
                ripple_set_ep1[user].append(ripple_set_ep1[user][-1])
                
            else:
                # sample a fixed-size 1-hop memory for each user
                # if size<memory, just use existing tripples repeatedly
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set_ep1[user].append((memories_h, memories_r, memories_t))
    
    # ripple_set_ep2           
    for user in user_history_dict_ep2:
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict_ep2[user]
            else:
                tails_of_last_hop = ripple_set_ep2[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])
            
            if len(memories_h) == 0:
                ripple_set_ep2[user].append(ripple_set_ep2[user][-1])
                
            else:
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set_ep2[user].append((memories_h, memories_r, memories_t))
        
    # ripple_set_ep3           
    for user in user_history_dict_ep3:
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict_ep3[user]
            else:
                tails_of_last_hop = ripple_set_ep3[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])
            
            if len(memories_h) == 0:
                ripple_set_ep3[user].append(ripple_set_ep3[user][-1])
                
            else:
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set_ep3[user].append((memories_h, memories_r, memories_t))
                
    return ripple_set_ep1, ripple_set_ep2, ripple_set_ep3


