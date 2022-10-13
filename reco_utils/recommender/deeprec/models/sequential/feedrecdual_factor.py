# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
from reco_utils.recommender.deeprec.models.sequential.sli_rec_dual import (
    SLI_RECModel_Dual,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import VecAttGRUCell
from reco_utils.recommender.deeprec.models.sequential.rnn_dien import dynamic_rnn as dynamic_rnn_dien
from reco_utils.recommender.deeprec.deeprec_utils import load_dict
from tensorflow.contrib.rnn import GRUCell, LSTMCell

__all__ = ["SASRecFeedDualModel_mask"]


class FeedDualModelFactor(SLI_RECModel_Dual):

    def _build_seq_graph(self):
        """The main function to create sasrec model.
        
        Returns:
            obj:the output of sasrec section.
        """
        with tf.name_scope('dfar'):
            self.seq = self.item_history_embedding
            self.seq = self.seq + self.position_embedding
            self.seq = self.seq + self.label_history_embedding

            self.mask = self.iterator.mask
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.pos_mask = tf.expand_dims(tf.cast(self.iterator.user_history, tf.float32), -1)
            self.neg_mask = tf.expand_dims(tf.cast(1 - self.iterator.user_history, tf.float32), -1)
            self.sequence_length = tf.reduce_sum(self.mask, 1)
            self.pos_label_embed = tf.expand_dims(self.label_lookup[1], 0)
            self.neg_label_embed = tf.expand_dims(self.label_lookup[0], 0)

            self.dropout_rate = 0.0
            self.num_blocks = 1
            self.hidden_units = self.item_embedding_dim

            self.is_training = True
            self.recent_k = 1

            self.seq *= tf.expand_dims(self.real_mask, -1)

            attention_output_pos, score_pos = self._attention_fcn(self.target_item_embedding + self.pos_label_embed, self.seq, 'Att_pos', False, return_alpha=True)
            attention_output_neg, score_neg = self._attention_fcn(self.target_item_embedding + self.neg_label_embed, self.seq, 'Att_neg', False, return_alpha=True)


            att_fea_pos = tf.reduce_sum(attention_output_pos, 1)
            att_fea_neg = tf.reduce_sum(attention_output_neg, 1)

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    self.seq, attn_score_f = self.multihead_attention(queries=self.normalize(self.seq),
                                                    keys=self.seq,
                                                    num_units=self.hidden_units,
                                                    num_heads=self.num_heads,
                                                    dropout_rate=self.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=True,
                                                    #  causality=False,
                                                    scope="self_attention",
                                                    is_mask_heads = True)


                    # Feed forward
                    self.seq = self.feedforward(self.normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate, is_training=self.is_training)
                    self.seq *= tf.expand_dims(self.real_mask, -1)

            self.seq = self.normalize(self.seq)

            # all 
            self.hist_embedding_mean = tf.reduce_sum(self.seq*tf.expand_dims(self.real_mask, -1), 1)/tf.reduce_sum(self.real_mask, 1, keepdims=True)
            self.seq_pos = self.seq * (self.pos_mask)
            self.seq_neg = self.seq * (self.neg_mask)


            for i in range(self.num_blocks):
                with tf.variable_scope("pos_num_blocks_%d" % i):

                    # Self-attention
                    self.seq_pos, attn_score = self.multihead_attention(queries=self.normalize(self.seq_pos),
                                                    keys=self.seq_pos,
                                                    num_units=self.hidden_units,
                                                    num_heads=self.num_heads,
                                                    dropout_rate=self.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=True,
                                                    #  causality=False,
                                                    scope="self_attention")


                    self.seq_pos = self.feedforward(self.normalize(self.seq_pos), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate, is_training=self.is_training)
                    self.seq_pos *= self.pos_mask

            self.seq_pos = self.normalize(self.seq_pos)


            for i in range(self.num_blocks):
                with tf.variable_scope("neg_num_blocks_%d" % i):

                    # Self-attention
                    self.seq_neg, attn_score = self.multihead_attention(queries=self.normalize(self.seq_neg),
                                                    keys=self.seq_neg,
                                                    num_units=self.hidden_units,
                                                    num_heads=self.num_heads,
                                                    dropout_rate=self.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=True,
                                                    #  causality=False,
                                                    scope="self_attention")


                    # Feed forward
                    self.seq_neg = self.feedforward(self.normalize(self.seq_neg), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate, is_training=self.is_training)
                    self.seq_neg *= self.neg_mask

            self.seq_neg = self.normalize(self.seq_neg)

            self.hist_embedding_mean_pos = tf.reduce_sum(self.seq_pos*tf.expand_dims(self.real_mask, -1), 1)/tf.reduce_sum(self.real_mask, 1, keepdims=True)
            self.hist_embedding_mean_neg = tf.reduce_sum(self.seq_neg*tf.expand_dims(self.real_mask, -1), 1)/tf.reduce_sum(self.real_mask, 1, keepdims=True)


        print("weight Long Bpr")
        model_output_pos = tf.concat([self.target_item_embedding, self.hist_embedding_mean, self.hist_embedding_mean_pos, att_fea_pos], -1)
        model_output_neg = tf.concat([self.target_item_embedding, self.hist_embedding_mean, self.hist_embedding_mean_neg, att_fea_neg], -1)

        
        return model_output_pos, model_output_neg, self.cosine_distance(self.hist_embedding_mean_pos, self.hist_embedding_mean_neg), self.cosine_distance(att_fea_pos, att_fea_neg), tf.reduce_mean(score_pos), tf.reduce_mean(score_neg), attn_score_f, self.seq
    def cosine_distance(self, pos_embed, neg_embed):
        x = tf.math.l2_normalize(pos_embed, axis=-1)
        y = tf.math.l2_normalize(neg_embed, axis=-1)
        return -tf.reduce_mean(x * y, axis=-1)
    def normalize(self, inputs, 
                  epsilon = 1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.
        
        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
          
        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
        
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta= tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
            
        return outputs

    def multihead_attention(self, queries, 
                            keys, 
                            num_units=None, 
                            num_heads=8, 
                            dropout_rate=0,
                            is_training=True,
                            causality=False,
                            scope="multihead_attention", 
                            reuse=None,
                            with_qk=False,
                            is_mask_heads = False):
        '''Applies multihead attention.
        
        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked. 
          因果关系：布尔值。 如果为true，则屏蔽引用未来的单位。
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
        Returns
          A 3d tensor with shape of (N, T_q, C)  
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]
            initial_units =  num_units
            num_units = num_units
            num_len = int(queries.shape.as_list()[1])

            # Linear projections

            Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            
            # Split and concat
            Q_ = tf.stack(tf.split(Q, num_heads, axis=2)) # (h_q, N, T_q, C/h) 
            K_ = tf.stack(tf.split(K, num_heads, axis=2)) # (h_k, N, T_k, C/h)
            # (h, N, C/h, T_k)
            V_ = tf.stack(tf.split(V, num_heads, axis=2)) # (h_k, N, T_k, C/h) (h, n, t_k, c/h) 
            
            # Multiplication
            Q_ = tf.transpose(Q_, perm=[1, 0, 2, 3]) # (N, h_q, T_q, C/h) 
            Q_ = tf.reshape(Q_, (-1, int(num_len * num_heads), int(num_units / num_heads))) 

            # print("Q", Q_.shape)
            K_ = tf.transpose(K_, perm=[1, 3, 2, 0]) # (N, C/h, T_k, h_k)
            K_ = tf.reshape(K_, (-1, int(num_units / num_heads), int(num_len * num_heads))) 
            # K_ = tf.transpose(K_, perm=[1, 0, 3, 2]) # (N, h_k, C/h, T_k)

            # print("K", K_.shape)
            V_ = tf.transpose(V_, perm=[1, 0, 2, 3]) # (N, h_k, T_k, C/h)
            V_ = tf.tile(tf.expand_dims(V_, 1), [1, num_heads, 1, 1, 1])# (N, h_q, T_k, C/h, h_k)

            outputs = tf.matmul(Q_, K_) # (N, h_q * T_q, T_k * h_k)
            # (N, h_q, T_q, T_k, h_k)
            # (N, h_q, T_k, C/h, h_k)
            # print(outputs.shape)
            # outputs = tf.tensordot(T_q, outputs, axes=(-1, 0))
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5) 
            
            # Key Masking
            key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) # (N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, num_heads, 1]) # (N, h_q, T_k)
            # print(key_masks.shape)

            key_masks = tf.tile(tf.expand_dims(key_masks, 2), [1, 1, tf.shape(keys)[1], 1]) # (N, h_q, T_q, T_k)
            # print(key_masks.shape)

            key_masks = tf.tile(tf.expand_dims(key_masks, -1), [1, 1, 1, 1, num_heads])    # (N, h_q, T_q, T_k, h_k)
            # key_masks = tf.transpose(key_masks, perm=[1, 0, 2, 3, 4]) 
            # (N, h_q, T_q, T_k, h_k)
            key_masks = tf.reshape(key_masks, (-1, int(num_len * num_heads), int(num_len * num_heads))) 
            # print(key_masks.shape)
            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (N, h_q, T_q, T_k, h_k)

      
            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :]) # (h_q * T_q, T_k * h_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (h_q * T_q, T_k * h_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) #(N, h_q * T_q, T_k * h_k)

                paddings = tf.ones_like(masks)*(-2**32+1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs) #(N, h_q * T_q, T_k * h_k)
      
            # Activation
            outputs = tf.nn.softmax(outputs) #(N, h_q * T_q, T_k * h_k)
            # outputs = tf.tensordot(T_v, outputs, axes=(-1, 0))

            # Query Masking
            query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1)) # (N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, 1), [1, num_heads, 1]) # (N, h_q, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, 1, tf.shape(keys)[1]]) # (N, h_q, T_q, T_k)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, 1, 1, num_heads]) # (N, h_q, T_q, T_k, h_k)
            query_masks = tf.reshape(query_masks, (-1, int(num_len * num_heads), int(num_len * num_heads))) 

            attn_score = outputs * query_masks #(N, h_q * T_q, T_k * h_k)
              
            # Dropouts
            #  outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            #(N, h_q, T_q, T_k, h_k)
            #(N, h_q, T_k, C/h, h_k)
            attn_score = tf.reshape(attn_score, (-1, num_heads, int(num_len), num_len, int(num_heads))) 
            attn_score = tf.transpose(attn_score, perm=[0, 1, 4, 2, 3])  #(N, h_q, h_k, T_q, T_k)
            # print("attn", attn_score.shape)
            # print("mask", self.head_mask.shape)
            if is_mask_heads:
                print("attn", attn_score.shape)
                print("mask", self.head_mask.shape)
                attn_score = attn_score * self.head_mask
            # else: 
            # Weighted sum
            outputs = tf.matmul(attn_score, V_) #(N, h_q, T_q, C/h, h_k)
            
            # Restore shape
            outputs = tf.transpose(outputs, perm=[0, 2, 1, 4, 3])  #(N, T_q, h_q, h_k, C/h)
            # outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
            outputs = tf.reshape(outputs, (-1, num_len, num_units * num_heads))  # (N, T_q, hidden)
            outputs = tf.layers.dense(outputs, initial_units, activation=None) # (N, T_q, C)

            # Residual connection
            outputs += queries
                  

        if with_qk: return Q,K
        else: return outputs, attn_score

    def feedforward(self, inputs, 
                    num_units=[2048, 512],
                    scope="multihead_attention", 
                    dropout_rate=0.2,
                    is_training=True,
                    reuse=None):
        '''Point-wise feed forward net.
        
        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            #  outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            #  outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            
            # Residual connection
            outputs += inputs
            
            # Normalize
            #outputs = normalize(outputs)
        
        return outputs


    def _attention_fcn(self, query, key_value, name, reuse, return_alpha=False):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item or cluster which is regarded as a query in attention operations.
            key_value (obj): The embedding of history items which is regarded as keys or values in attention operations.
            name (obj): The name of variable W 
            reuse (obj): Reusing variable W in query operation 
            return_alpha (obj): Returning attention weights

        Returns:
            output (obj): Weighted sum of value embedding.
            att_weights (obj):  Attention weights
        """
        with tf.variable_scope("attention_fcn"+str(name), reuse=reuse):
            query_size = query.shape[-1].value
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            attention_mat = tf.get_variable(
                name="attention_mat"+str(name),
                shape=[key_value.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(key_value, attention_mat, [[2], [0]])

            if query.shape.ndims != att_inputs.shape.ndims:
                queries = tf.reshape(
                    tf.tile(query, [1, tf.shape(att_inputs)[1]]), tf.shape(att_inputs)
                )
            else:
                queries = query

            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )
            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, self.hparams.att_fcn_layer_sizes, scope="att_fcn"
            )
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = key_value * tf.expand_dims(att_weights, -1)
            if not return_alpha:
                return output
            else:
                return output, att_fnc_output


