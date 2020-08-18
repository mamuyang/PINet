
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
            with tf.name_scope('prediction_A'):
                self.pred_A = self.prediction_A(self.num_items_A, encoder_state_A, self.keep_prob)
            with tf.name_scope('prediction_B'):
                self.pred_B = self.prediction_B(self.num_items_B, encoder_state_B, self.keep_prob)            
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
    
    def prediction_A(self, num_items_A, encoder_state_A, keep_prob):
        with tf.variable_scope('prediction_A'):
            dropout_output = tf.nn.dropout(encoder_state_A[-1], keep_prob)#dropout_output=[batch_size,hidden_size]
            pred_A = tf.layers.dense(dropout_output, num_items_A, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))#pred_A=[batch_size,num_items_A]
            print(pred_A)
        return pred_A
    
    def prediction_B(self, num_items_B, encoder_state_B, keep_prob):
        with tf.variable_scope('prediction_B'):
            dropout_output = tf.nn.dropout(encoder_state_B[-1], keep_prob)#dropout_output=[batch_size,hidden_size]
            pred_B = tf.layers.dense(dropout_output, num_items_B, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))#pred_B=[batch_size,num_items_B]
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