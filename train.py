import tensorflow as tf
import numpy as np
from model import RippleNet
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report 
PLOT = False

def train(args, data_info, show_loss):

    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    model = RippleNet(args, n_entity, n_relation)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epoch):
            # training
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]-args.batch_size:
                    
                _, loss= model.train(sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
                
                #print (temp)
                start += args.batch_size
                if show_loss:
                    if start / (train_data.shape[0]-args.batch_size) >0.999:
                        print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))

            # evaluation
            if PLOT:
                test_auc, test_acc = evaluation(sess, args, model, test_data, ripple_set, args.batch_size,step,args.n_epoch)
                print('epoch %d   test auc: %.4f  acc: %.4f'
                      % (step,test_auc, test_acc))
            else: 
                train_auc, train_acc = evaluation(sess, args, model, train_data, ripple_set, args.batch_size)
                eval_auc, eval_acc = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size)
                test_auc, test_acc = evaluation(sess, args, model, test_data, ripple_set, args.batch_size,step,args.n_epoch)
                print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                      % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))   
    print('test test ...')

def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    for i in range(args.n_hop):
        #feed_dict[model.memories_h_ep1[i]] = temp
        feed_dict[model.memories_h_ep1[i]] = [ripple_set[0][user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_h_ep2[i]] = [ripple_set[0][user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_h_ep3[i]] = [ripple_set[0][user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r_ep1[i]] = [ripple_set[1][user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_r_ep2[i]] = [ripple_set[1][user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_r_ep3[i]] = [ripple_set[1][user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t_ep1[i]] = [ripple_set[2][user][i][2] for user in data[start:end, 0]]  
        feed_dict[model.memories_t_ep2[i]] = [ripple_set[2][user][i][2] for user in data[start:end, 0]]  
        feed_dict[model.memories_t_ep3[i]] = [ripple_set[2][user][i][2] for user in data[start:end, 0]]  
    return feed_dict

def evaluation(sess, args, model, data, ripple_set, batch_size, step, epoch):
    '''
    start = 0
    auc_list = []
    acc_list = []
    while start < data.shape[0]-batch_size:
    
        auc, acc = model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list))
    '''
    from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, roc_auc_score      

    start = 0
    labels_list = []
    scores_list = []
    patients_list = []
    
    while start < data.shape[0]-batch_size:
        labels, scores= model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        for i in labels: labels_list.append(i)
        for i in scores: scores_list.append(i)
        for i in data[start: start+batch_size,0]:patients_list.append(i) # data is array, column represents patid, item, label, episode 
        start += batch_size
    auc = roc_auc_score(y_true=labels_list, y_score=scores_list)
    predictions = [1 if i >= 0.5 else 0 for i in scores_list]
    acc = np.mean(np.equal(predictions, labels_list))
    '''
    f = open('label_score.csv','w')
    if acc > 0.7:
        for lab, sco, pred in zip(labels_list, scores_list, predictions):
            f.write(str(lab)+','+str(sco)+','+str(pred)+'\n')
    f.close()
    '''
    print(classification_report(labels_list,predictions,digits=3))

    #---------p-r curve------
    if step == epoch-1:
        with open('D:/research/ripplenet/RippleNet1201/src/pr_curve/label_score.txt','w') as infile:
            for l, s in zip(labels_list, scores_list):
                infile.write(str(l)+','+str(s)+'\n')
    
    #---------topk(with patid)---------
    if step == epoch-1:
        with open('D:/research/ripplenet/RippleNet1201/src/topk/topk_results.txt','w') as infile:
            for l, s, p in zip(labels_list, scores_list, patients_list):
                infile.write(str(l)+','+str(s)+','+str(p)+'\n') 
    return auc, acc






