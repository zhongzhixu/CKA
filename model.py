import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.contrib import rnn
from tensorflow import keras

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM

class RippleNet(object):
    def __init__(self, args, n_entity, n_relation):
        self._parse_args(args, n_entity, n_relation)
        self._build_inputs()
        self._build_embeddings()
        self._rnn_variables()
        self._build_model()
        self._build_loss()
        self._build_train()
        

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory #size of ripple set for each hop
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops
        self.dimhidden = args.dimhidden
        self.diminput = args.diminput
        self.dimoutput = args.dimoutput
        self.nsteps = args.nsteps
        self.is_sequencial = args.is_sequencial
        
    def _build_inputs(self):
        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name="labels")
        self.memories_h_ep1, self.memories_h_ep2, self.memories_h_ep3  = [], [], []
        self.memories_r_ep1, self.memories_r_ep2, self.memories_r_ep3  = [], [], []
        self.memories_t_ep1, self.memories_t_ep2, self.memories_t_ep3  = [], [], []

        '''
        self.memories_h = []
        self.memories_r = []
        self.memories_t = []
        '''
        
        for hop in range(self.n_hop):
            self.memories_h_ep1.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_ep1_" + str(hop)))
            self.memories_h_ep2.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_ep2_" + str(hop)))            
            self.memories_h_ep3.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_ep3_" + str(hop)))           
            
            self.memories_r_ep1.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_ep1_" + str(hop)))
            self.memories_r_ep2.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_ep2_" + str(hop)))
            self.memories_r_ep3.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_ep3_" + str(hop)))
            
            self.memories_t_ep1.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_ep1_" + str(hop)))
            self.memories_t_ep2.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_ep2_" + str(hop)))
            self.memories_t_ep3.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_ep3_" + str(hop)))

    def _build_embeddings(self):
        self.entity_emb_matrix = tf.get_variable(name="entity_emb_matrix", dtype=tf.float32,
                                                 shape=[self.n_entity, self.dim],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        self.relation_emb_matrix = tf.get_variable(name="relation_emb_matrix", dtype=tf.float32,
                                                   shape=[self.n_relation, self.dim, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())
        #tf.get_variable_scope().reuse_variables()
    def _rnn_variables(self):
        self.W = {"h1_hop0" : tf.Variable(tf.random_normal([self.diminput, self.dimhidden]), dtype=tf.float32), 
                  "h1_hop1" : tf.Variable(tf.random_normal([self.diminput, self.dimhidden]), dtype=tf.float32), 
                  "h2_hop0" : tf.Variable(tf.random_normal([self.dimhidden, self.dimoutput]), dtype=tf.float32),
                  "h2_hop1" : tf.Variable(tf.random_normal([self.dimhidden, self.dimoutput]), dtype=tf.float32)}

        self.b = {"b1_hop0" : tf.Variable(tf.random_normal([self.dimhidden]), dtype=tf.float32), 
                  "b1_hop1" : tf.Variable(tf.random_normal([self.dimhidden]), dtype=tf.float32), 
                  "b2_hop0" : tf.Variable(tf.random_normal([self.dimoutput]), dtype=tf.float32),
                  "b2_hop1" : tf.Variable(tf.random_normal([self.dimoutput]), dtype=tf.float32)}

    def _build_model(self):
        '''
        This fuction is used to calculate score 
        and then will be fed into loss
        '''
        # transformation matrix for updating item embeddings at the end of each hop
        self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.dim, self.dim], dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())
        # [batch size, dim]
        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        self.h_emb_list_ep1, self.h_emb_list_ep2, self.h_emb_list_ep3  = [],[],[]
        self.r_emb_list_ep1, self.r_emb_list_ep2, self.r_emb_list_ep3 = [],[],[]
        self.t_emb_list_ep1, self.t_emb_list_ep2, self.t_emb_list_ep3 = [],[],[]
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            self.h_emb_list_ep1.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h_ep1[i]))
            self.h_emb_list_ep2.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h_ep2[i]))
            self.h_emb_list_ep3.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h_ep3[i]))
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h_ep1[i]))
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h_ep2[i]))
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h_ep3[i]))
            
            # [batch size, n_memory, dim, dim]
            self.r_emb_list_ep1.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r_ep1[i]))
            self.r_emb_list_ep2.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r_ep2[i]))
            self.r_emb_list_ep3.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r_ep3[i]))
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r_ep1[i]))
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r_ep2[i]))
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r_ep3[i]))

            # [batch size, n_memory, dim]
            self.t_emb_list_ep1.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t_ep1[i]))
            self.t_emb_list_ep2.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t_ep2[i]))
            self.t_emb_list_ep3.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t_ep3[i]))
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t_ep1[i]))
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t_ep2[i]))
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t_ep3[i]))
            
        o_list = self._key_addressing()
        self.scores = tf.squeeze(self.predict(self.item_embeddings, o_list))
        self.scores_normalized = tf.sigmoid(self.scores)

    def _key_addressing(self):
        '''
        Note that n_memory_ep1=n_memory_ep2=n_memory_ep3=n_memory
        '''
        o_list = []
        lstm_cell_0 = tf.nn.rnn_cell.BasicLSTMCell(self.dimhidden,forget_bias=1.0)
        lstm_cell_1 = tf.nn.rnn_cell.BasicRNNCell(self.dimhidden)
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            #h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)
            h_expanded_ep1 = tf.expand_dims(self.h_emb_list_ep1[hop], axis=3)
            h_expanded_ep2 = tf.expand_dims(self.h_emb_list_ep2[hop], axis=3)
            h_expanded_ep3 = tf.expand_dims(self.h_emb_list_ep3[hop], axis=3)            

            # [batch_size, n_memory, dim]
            #Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)
            Rh_ep1 = tf.squeeze(tf.matmul(self.r_emb_list_ep1[hop], h_expanded_ep1), axis=3)
            Rh_ep2 = tf.squeeze(tf.matmul(self.r_emb_list_ep2[hop], h_expanded_ep2), axis=3)
            Rh_ep3 = tf.squeeze(tf.matmul(self.r_emb_list_ep3[hop], h_expanded_ep3), axis=3)
            
            # [batch_size, dim, 1]
            v = tf.expand_dims(self.item_embeddings, axis=2)

            # [batch_size, n_memory]
            #self.probs = tf.squeeze(tf.matmul(Rh, v), axis=2)
            self.probs_ep1 = tf.squeeze(tf.matmul(Rh_ep1, v), axis=2)
            self.probs_ep2 = tf.squeeze(tf.matmul(Rh_ep2, v), axis=2)
            self.probs_ep3 = tf.squeeze(tf.matmul(Rh_ep3, v), axis=2)
            
            # [batch_size, n_memory]
            #probs_normalized = tf.nn.softmax(self.probs)
            probs_normalized_ep1 = tf.nn.softmax(self.probs_ep1)
            probs_normalized_ep2 = tf.nn.softmax(self.probs_ep2)
            probs_normalized_ep3 = tf.nn.softmax(self.probs_ep3)

            # newly added: sequencial information considered
            # [batch_size, n_memory, 1]
            # probs_expanded = tf.expand_dims(probs_normalized, axis=2)               
            probs_expanded_ep1 = tf.expand_dims(probs_normalized_ep1, axis=2)
            probs_expanded_ep2 = tf.expand_dims(probs_normalized_ep2, axis=2)
            probs_expanded_ep3 = tf.expand_dims(probs_normalized_ep3, axis=2)
            
            # RNN
            # [batch size, n_memory, dim] * [batch_size, n_memory, 1]
            #NOTE: Here is multiply, not matmult, so 
            #the size before reduce_sum         :[batch size, n_memory, dim]
            #the size after reduce_sum(axis=1)  :[batch size, dim]
            x_ep1 = tf.reduce_sum(self.t_emb_list_ep1[hop] * probs_expanded_ep1, axis = 1)
            x_ep2 = tf.reduce_sum(self.t_emb_list_ep2[hop] * probs_expanded_ep2, axis = 1)
            x_ep3 = tf.reduce_sum(self.t_emb_list_ep3[hop] * probs_expanded_ep3, axis = 1)
            #print (x_ep1.get_shape().as_list())
            
            if self.is_sequencial is True:
                x = tf.concat([x_ep1, x_ep2, x_ep3], axis = 1)
                x = tf.reshape(x,[-1, self.nsteps, self.dim])
                #########
                x = tf.transpose(x,[1,0,2])
                x = tf.reshape(x,[-1,self.diminput])
                #tf.cast(x, tf.float32)
                X = tf.matmul(x,self.W["h1"+'_hop'+str(hop)])+self.b["b1"+'_hop'+str(hop)]
                X = tf.split(X,self.nsteps,0)
                
                if hop == 0: LSTM_O,LSTM_S = rnn.static_rnn(lstm_cell_0, X,dtype=tf.float32)
                if hop == 1: LSTM_O,LSTM_S = rnn.static_rnn(lstm_cell_1, X,dtype=tf.float32)
                
                O = tf.matmul(LSTM_O[-1],self.W["h2"+'_hop'+str(hop)])+self.b["b2"+'_hop'+str(hop)] # for sure you will introduce new variables (for rnn)
            
            else:
                O=x_ep1+x_ep2+x_ep3 
            
            #update v
            self.item_embeddings = self.update_item_embedding(self.item_embeddings, O) #refere to using o to replace v in the eq(4)
            o_list.append(O)
            #newly added: over 
            
            
            '''
            # [batch_size, dim]
            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)
            
            #update v
            self.item_embeddings = self.update_item_embedding(self.item_embeddings, o)
            o_list.append(o)
            '''
        return o_list

    def update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = tf.matmul(o, self.transform_matrix)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # [batch_size]
        scores = tf.reduce_sum(item_embeddings * y, axis=1)
        return scores

    def _build_loss(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

        self.kge_loss = 0
        for hop in range(self.n_hop):
            # [batch size, n_memory, dim] to [batch size, n_memory, 1, dim]
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
            # [batch size, n_memory, dim] to [batch size, n_memory, dim, 1]
            t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
            self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
        self.kge_loss = -self.kge_weight * self.kge_loss

        self.l2_loss = 0
        for hop in range(self.n_hop):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
            if self.item_update_mode == "replace nonlinear" or self.item_update_mode == "plus nonlinear":
                self.l2_loss += tf.nn.l2_loss(self.transform_matrix)
        self.l2_loss = self.l2_weight * self.l2_loss

        self.loss = self.base_loss + self.kge_loss + self.l2_loss
        #self.loss = self.base_loss
        
    def _build_train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, clip_norm=5)
                     for gradient in gradients]
        self.optimizer = optimizer.apply_gradients(zip(gradients, variables))
        '''

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        return labels, scores
    
        


    
    