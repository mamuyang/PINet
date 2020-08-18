
# coding: utf-8

# In[ ]:

import os
import random
import numpy as np
import tensorflow as tf


# In[ ]:

random.seed(1)
np.random.seed(1)


# In[ ]:

class FilterCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, member_embedding, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None):
        super(FilterCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._member_embedding = member_embedding
        self._activation = activation or tf.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer 
        
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units    
    
    def call(self, inputs, state):#inputs=[batch_size,hidden_size+hidden_size] , state=[batch_size,self._num_units] 
        inputs_A, inputs_T = tf.split(inputs, num_or_size_splits=2, axis=1)#inputs_A=[batch_size,hidden_size]，inputs_T=[batch_size,hidden_size]
        if self._kernel_initializer is None:
            self._kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        if self._bias_initializer is None:
            self._bias_initializer = tf.constant_initializer(1.0)       
        with tf.variable_scope('gate'):#sigmoid([i_A|i_T|s_(t-1)]*[W_fA;W_fT;U_fS]+emb*V_f+b_f)
            self.W_f = tf.get_variable(dtype=tf.float32, name='W_f', shape=[inputs.get_shape()[-1].value+state.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer)         
            self.V_f = tf.get_variable(dtype=tf.float32, name='V_f', shape=[self._member_embedding.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer)
            self.b_f = tf.get_variable(dtype=tf.float32, name='b_f', shape=[self._num_units,], initializer=self._bias_initializer)
            u = tf.matmul(self._member_embedding, self.V_f)#u=[num_members,self._num_units]
            f = tf.concat([inputs, state], axis=-1)#f=[batch_size,hidden_size+hidden_size+self._num_units]
            f = tf.matmul(f, self.W_f)#f=[batch_size,self._num_items]
            f = f+self.b_f#f=[batch_size,self._num_items]
            f = tf.expand_dims(f, axis=1)#f=[batch_size,1,self._num_items]
            f = tf.tile(f, [1,u.get_shape()[0].value,1])#f=[batch_size,num_members,self._num_items]
            f = f+u#f=[batch_size,num_members,self._num_items]
            f = tf.sigmoid(f)
        with tf.variable_scope('candidate'):#tanh([i_A|s_(t-1)]*[W_sA;U_sS]+emb*V_s+b_s)
            self.W_s = tf.get_variable(dtype=tf.float32, name='W_s', shape=[inputs_A.get_shape()[-1].value+state.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer) 
            self.V_s = tf.get_variable(dtype=tf.float32, name='V_s', shape=[self._member_embedding.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer)
            self.b_s = tf.get_variable(dtype=tf.float32, name='b_s', shape=[self._num_units,], initializer=self._bias_initializer)
            _u = tf.matmul(self._member_embedding, self.V_s)#_u=[num_members,self._num_units]
            _s = tf.concat([inputs_A, state], axis=-1)#_s=[batch_size,hidden_size+self._num_units]
            _s = tf.matmul(_s, self.W_s)#_s=[batch_size,self._num_items]
            _s = _s+self.b_s#_s=[batch_size,self._num_items]
            _s = tf.expand_dims(_s, axis=1)#_s=[batch_size,1,self._num_items]
            _s = tf.tile(_s, [1,_u.get_shape()[0].value,1])#_s=[batch_size,num_members,self._num_items]
            _s = _s+u#_s=[batch_size,num_members,self._num_items]
            _s = self._activation(_s)
        state = tf.expand_dims(state, axis=1)#state=[batch_size,1,self._num_units]
        state = tf.tile(state, [1,self._member_embedding.get_shape()[0].value,1])#state=[batch_size,num_members,self._num_items]
        new_s = f*state+(1-f)*_s#new_s=[batch_size,num_members,self._num_items]
        new_s = tf.reduce_mean(new_s, axis=1)#new_s=[batch_size,self._num_items]
        return new_s, new_s


# In[ ]:

class PiNet():
    def __init__(self, num_items_A, num_items_B, num_members=4, embedding_size=100, hidden_size=100, num_layers=1, gpu='0'):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.num_items_A = num_items_A
        self.num_items_B = num_items_B
        self.num_members = num_members
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1)
            with tf.name_scope('inputs'):
                self.seq_A, self.seq_B, self.len_A, self.len_B, self.pos_A, self.pos_B, self.target_A, self.target_B, self.learning_rate, self.keep_prob = self.get_inputs()
            with tf.name_scope('encoder_A'):
                encoder_output_A,encoder_state_A = self.encoder_A(self.num_items_A, self.seq_A, self.len_A, self.embedding_size, self.hidden_size, self.num_layers, self.keep_prob)                           
            with tf.name_scope('encoder_B'):
                encoder_output_B,encoder_state_B = self.encoder_B(self.num_items_B, self.seq_B, self.len_B, self.embedding_size, self.hidden_size, self.num_layers, self.keep_prob)
            with tf.name_scope('filter_B'):
                filter_output_B,filter_state_B = self.filter_B(encoder_output_A, encoder_output_B, self.len_B, self.pos_B, self.num_members, self.embedding_size, self.hidden_size, self.num_layers, self.keep_prob)
            with tf.name_scope('transfer_B'):
                transfer_output_B,transfer_state_B = self.transfer_B(filter_output_B, self.len_B, self.hidden_size, self.num_layers, self.keep_prob)
            with tf.name_scope('prediction_A'):
                self.pred_A = self.prediction_A(self.num_items_A, transfer_state_B, encoder_state_A, self.keep_prob)
            with tf.name_scope('filter_A'):
                filter_output_A,filter_state_A = self.filter_A(encoder_output_B, encoder_output_A, self.len_A, self.pos_A, self.num_members, self.embedding_size, self.hidden_size, self.num_layers, self.keep_prob)
            with tf.name_scope('transfer_A'):
                transfer_output_A,transfer_state_A = self.transfer_A(filter_output_A, self.len_A, self.hidden_size, self.num_layers, self.keep_prob)
            with tf.name_scope('prediction_B'):
                self.pred_B = self.prediction_B(self.num_items_B, transfer_state_A, encoder_state_B, self.keep_prob)            
            with tf.name_scope('loss'):
                self.loss = self.cal_loss(self.target_A, self.pred_A, self.target_B, self.pred_B)
            with tf.name_scope('optimizer'):
                self.train_op = self.optimizer(self.loss, self.learning_rate)
        
    def get_inputs(self):
        seq_A = tf.placeholder(dtype=tf.int32, shape=[None,None], name='seq_A')
        seq_B = tf.placeholder(dtype=tf.int32, shape=[None,None], name='seq_B')
        len_A = tf.placeholder(dtype=tf.int32, shape=[None,], name='len_A')
        len_B = tf.placeholder(dtype=tf.int32, shape=[None,], name='len_B')
        pos_A = tf.placeholder(dtype=tf.int32, shape=[None,None,2], name='pos_A')
        pos_B = tf.placeholder(dtype=tf.int32, shape=[None,None,2], name='pos_B')
        target_A = tf.placeholder(dtype=tf.int32, shape=[None,], name='target_A') 
        target_B = tf.placeholder(dtype=tf.int32, shape=[None,], name='target_B') 
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        return seq_A, seq_B, len_A, len_B, pos_A, pos_B, target_A, target_B, learning_rate, keep_prob
    
    def get_gru_cell(self, hidden_size, keep_prob):
        gru_cell = tf.contrib.rnn.GRUCell(hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob, state_keep_prob=keep_prob)  
        return gru_cell
    
    def get_filter_cell(self, hidden_size, member_embedding, keep_prob):
        filter_cell = FilterCell(hidden_size, member_embedding)
        filter_cell = tf.contrib.rnn.DropoutWrapper(filter_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob, state_keep_prob=keep_prob)  
        return filter_cell

    def encoder_A(self, num_items_A, seq_A, len_A, embedding_size, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('encoder_A'):
            embedding_matrix_A = tf.get_variable(dtype=tf.float32, name='embedding_matrix_A', shape=[num_items_A,embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(embedding_matrix_A)
            embbed_seq_A = tf.nn.embedding_lookup(embedding_matrix_A, seq_A)#embbed_seq_A=[batch_size,timestamp_A,embedding_size]
            encoder_cell_A = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob) for _ in range(num_layers)])
            encoder_output_A,encoder_state_A = tf.nn.dynamic_rnn(encoder_cell_A, embbed_seq_A, sequence_length=len_A, dtype=tf.float32)#encoder_output_A=[batch_size,timestamp_A,hidden_size], encoder_state_A=([batch_size,hidden_size]*num_layers)       
            print(encoder_output_A)
            print(encoder_state_A)
        return encoder_output_A,encoder_state_A
    
    def encoder_B(self, num_items_B, seq_B, len_B, embedding_size, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('encoder_B'):
            embedding_matrix_B = tf.get_variable(dtype=tf.float32, name='embedding_matrix_B', shape=[num_items_B,embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))   
            print(embedding_matrix_B)
            embbed_seq_B = tf.nn.embedding_lookup(embedding_matrix_B, seq_B)#embbed_seq_B=[batch_size,timestamp_B,embedding_size]
            encoder_cell_B = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob) for _ in range(num_layers)])
            encoder_output_B,encoder_state_B = tf.nn.dynamic_rnn(encoder_cell_B, embbed_seq_B, sequence_length=len_B, dtype=tf.float32)#encoder_output_B=[batch_size,timestamp_B,hidden_size], encoder_state_B=([batch_size,hidden_size]*num_layers)    
            print(encoder_output_B)
            print(encoder_state_B)
        return encoder_output_B,encoder_state_B
    
    def filter_B(self, encoder_output_A, encoder_output_B, len_B, pos_B, num_members, embedding_size, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('filter_B'):
            zero_state = tf.zeros(dtype=tf.float32, shape=(tf.shape(encoder_output_A)[0],1,tf.shape(encoder_output_A)[-1]))#zero_state=[batch_size,1,hidden_size]                                 
            print(zero_state)
            encoder_output = tf.concat([zero_state,encoder_output_A], axis=1)#encoder_output=[batch_size,timestamp_A+1,hidden_size]
            print(encoder_output)
            select_output_A = tf.gather_nd(encoder_output,pos_B)#select_output_A=[batch_size,timestamp_B,hidden_size]
            print(select_output_A)
            filter_input_B = tf.concat([encoder_output_B,select_output_A], axis=-1)#filter_input_B=[batch_size,timestamp_B,hidden_size+hidden_size]
            print(filter_input_B)
            member_embedding_B = tf.get_variable(dtype=tf.float32, name='member_embedding_B', shape=[num_members,embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(member_embedding_B)
            filter_cell_B = tf.contrib.rnn.MultiRNNCell([self.get_filter_cell(hidden_size, member_embedding_B, keep_prob) for _ in range(num_layers)])
            filter_output_B,filter_state_B = tf.nn.dynamic_rnn(filter_cell_B, filter_input_B, sequence_length=len_B, dtype=tf.float32)#filter_output_B=[batch_size,timestamp_B,hidden_size]，filter_state_B=[batch_size,hidden_size]            
            print(filter_output_B)
            print(filter_state_B)
        return filter_output_B,filter_state_B
        
    def transfer_B(self, filter_output_B, len_B, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('transfer_B'):
            transfer_cell_B = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob) for _ in range(num_layers)])
            transfer_output_B,transfer_state_B = tf.nn.dynamic_rnn(transfer_cell_B, filter_output_B, sequence_length=len_B, dtype=tf.float32)#transfer_output_B=[batch_size,timestamp_B,hidden_size], transfer_state_B=([batch_size,hidden_size]*num_layers)     
            print(transfer_output_B)
            print(transfer_state_B)
        return transfer_output_B,transfer_state_B
    
    def prediction_A(self, num_items_A, transfer_state_B, encoder_state_A, keep_prob):
        with tf.variable_scope('prediction_A'):
            concat_output = tf.concat([transfer_state_B[-1],encoder_state_A[-1]], axis=-1)                                                      
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output, keep_prob)#concat_output=[batch_size,hidden_size+hidden_size]
            pred_A = tf.layers.dense(concat_output, num_items_A, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))#pred_A=[batch_size,num_items_A]
            print(pred_A)
        return pred_A
    
    def filter_A(self, encoder_output_B, encoder_output_A, len_A, pos_A, num_members, embedding_size, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('filter_A'):
            zero_state = tf.zeros(dtype=tf.float32, shape=(tf.shape(encoder_output_B)[0],1,tf.shape(encoder_output_B)[-1]))#zero_state=[batch_size,1,hidden_size]                                     
            print(zero_state)
            encoder_output = tf.concat([zero_state,encoder_output_B], axis=1)#encoder_output=[batch_size,timestamp_B+1,hidden_size]
            print(encoder_output)
            select_output_B = tf.gather_nd(encoder_output,pos_A)#select_output_B=[batch_size,timestamp_A,hidden_size]
            print(select_output_B)
            filter_input_A = tf.concat([encoder_output_A,select_output_B], axis=-1)#filter_input_A=[batch_size,timestamp_A,hidden_size+hidden_size]
            print(filter_input_A)
            member_embedding_A = tf.get_variable(dtype=tf.float32, name='member_embedding_A', shape=[num_members,embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(member_embedding_A)
            filter_cell_A = tf.contrib.rnn.MultiRNNCell([self.get_filter_cell(hidden_size, member_embedding_A, keep_prob) for _ in range(num_layers)])
            filter_output_A,filter_state_A = tf.nn.dynamic_rnn(filter_cell_A, filter_input_A, sequence_length=len_A, dtype=tf.float32)#filter_output_A=[batch_size,timestamp_A,hidden_size]，filter_state_A=[batch_size,hidden_size]            
            print(filter_output_A)
            print(filter_state_A)
        return filter_output_A,filter_state_A
        
    def transfer_A(self, filter_output_A, len_A, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('transfer_A'):
            transfer_cell_A = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob) for _ in range(num_layers)])
            transfer_output_A,transfer_state_A = tf.nn.dynamic_rnn(transfer_cell_A, filter_output_A, sequence_length=len_A, dtype=tf.float32)#transfer_output_A=[batch_size,timestamp_A,hidden_size], transfer_state_A=([batch_size,hidden_size]*num_layers)     
            print(transfer_output_A)
            print(transfer_state_A)
        return transfer_output_A,transfer_state_A
    
    def prediction_B(self, num_items_B, transfer_state_A, encoder_state_B, keep_prob):
        with tf.variable_scope('prediction_B'):
            concat_output = tf.concat([transfer_state_A[-1],encoder_state_B[-1]], axis=-1)
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output, keep_prob)#concat_output=[batch_size,hidden_size+hidden_size]
            pred_B = tf.layers.dense(concat_output, num_items_B, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))#pred_B=[batch_size,num_items_B]
            print(pred_B)
        return pred_B
    
    def cal_loss(self, target_A, pred_A, target_B, pred_B):
        loss_A = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_A, logits=pred_A)
        loss_A = tf.reduce_mean(loss_A, name='loss_A')
        loss_B = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_B, logits=pred_B)
        loss_B = tf.reduce_mean(loss_B, name='loss_B')
        loss = loss_A+loss_B
        return loss
    
    def optimizer(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)                                
        gradients = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad,-5.,5.),var) for grad,var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        return train_op