# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
from reco_utils.recommender.deeprec.models.sequential.sli_rec import (
    SLI_RECModel,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import VecAttGRUCell
from reco_utils.recommender.deeprec.models.sequential.rnn_dien import dynamic_rnn as dynamic_rnn_dien
from reco_utils.recommender.deeprec.deeprec_utils import load_dict
from tensorflow.contrib.rnn import GRUCell, LSTMCell

__all__ = ["SASRecNegModel"]


class SASRecNegBPRModel(SLI_RECModel):

#      def _build_graph(self):
        #  """The main function to create sequential models.
        #
        #  Returns:
        #      obj:the prediction score make by the model.
        #  """
        #  hparams = self.hparams
        #  self.keep_prob_train = 1 - np.array(hparams.dropout)
        #  self.keep_prob_test = np.ones_like(hparams.dropout)
        #
        #  self.embedding_keep_prob_train = 1.0 - hparams.embedding_dropout
        #  if hparams.test_dropout:
        #      self.embedding_keep_prob_test = 1.0 - hparams.embedding_dropout
        #  else:
        #      self.embedding_keep_prob_test = 1.0
        #
        #  with tf.variable_scope("sequential") as self.sequential_scope:
        #      self._build_embedding()
        #      self._lookup_from_embedding()
        #      #  model_output = self._build_seq_graph()
        #      #  logit = self._fcn_net(model_output, hparams.layer_sizes, scope="logit_fcn")
        #      # for inner product
        #      logit = self._build_seq_graph()
        #      self._add_norm()
        #      return logit
#
    def _build_seq_graph(self):
        """The main function to create sasrec model.
        
        Returns:
            obj:the output of sasrec section.
        """
        with tf.name_scope('sasrec'):
            self.seq = self.item_history_embedding
            self.seq = self.seq + self.position_embedding
            self.mask = self.iterator.mask
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.sequence_length = tf.reduce_sum(self.mask, 1)
            # print() # label
            # relation_label = self.iterator.user_history
            # pos_relation =  tf.constant(-1, shape=[relation_label.shape[0], relation_label.shape[1]], dtype=tf.float32)
            # neg_relation = tf.constant(1, shape=[1, 20], dtype=tf.float32)
            # relation_label = tf.where(tf.equal(relation_label, 0), pos_relation, neg_relation)
            # self.relation_mask = tf.matmul(tf.expand_dims(relation_label, 2), tf.expand_dims(relation_label, 1))
            # self.relation_mask = 
            # self.relation_mask = (tf.cast(self.relation_mask, tf.float32))
            
            #  attention_output = self._attention_fcn(self.target_item_embedding, hist_input)
            #  att_fea = tf.reduce_sum(attention_output, 1)
            # hyper-parameters
            self.dropout_rate = 0.0
            self.num_blocks = 1
            self.hidden_units = self.item_embedding_dim
            self.num_heads = 1
            self.is_training = True
            #  self.recent_k = 5
            self.recent_k = 1
             # b, d
            # Dropout
            #  self.seq = tf.layers.dropout(self.seq,
                                         #  rate=self.dropout_rate,
                                         #  training=tf.convert_to_tensor(self.is_training))
            self.seq *= tf.expand_dims(self.real_mask, -1)

            # previous atte
            # attention_output, alphas = self._attention_fcn(self.target_item_embedding, self.seq, 'Att', False, return_alpha=True)
            # att_fea = tf.reduce_sum(attention_output, 1)

            # Build blocks
            # 堆叠的注意力，参数为2
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq, attn_score_relation = self.multihead_attention(queries=self.normalize(self.seq),
                                                    keys=self.seq,
                                                    num_units=self.hidden_units,
                                                    num_heads=self.num_heads,
                                                    dropout_rate=self.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=True,
                                                    #  causality=False,
                                                    scope="self_attention")

                    #  self.seq, alphas = self._attention_fcn(self.target_item_embedding, self.seq, 'Att', False, return_alpha=True)
                    #  self.seq, alphas = self._attention_fcn(self.target_item_embedding, self.normalize(self.seq), 'Att', False, return_alpha=True)

                    # synthesizer
                #    [ self.seq = self.synthesizer(queries=self.normalize(self.seq),
                #     #  self.seq = self.synthesizer(queries=self.seq,
                #                                    keys=self.seq,
                #                                     alphas = alphas, # add
                #                                    num_units=self.hidden_units,
                #                                    num_heads=self.num_heads,
                #                                    dropout_rate=self.dropout_rate,
                #                                    is_training=self.is_training,
                #                                    #  causality=True,
                #                                    causality=False,
                #                                    scope="self_attention")]

                    # Feed forward
                    self.seq = self.feedforward(self.normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate, is_training=self.is_training)
                    self.seq *= tf.expand_dims(self.real_mask, -1)

            self.seq = self.normalize(self.seq)

            # all 
            #  self.hist_embedding_sum = tf.reduce_sum(self.seq*tf.expand_dims(self.real_mask, -1), 1)
            self.hist_embedding_mean = tf.reduce_sum((self.seq)*tf.expand_dims(self.real_mask, -1), 1)/tf.reduce_sum(self.real_mask, 1, keepdims=True)
            #  import pdb; pdb.set_trace()
            #  seq_shape = self.seq.get_shape()
            #  self.hist_embedding_concat = tf.reshape(self.seq*tf.expand_dims(self.real_mask, -1), [tf.shape(self.seq)[0],seq_shape[-1]*seq_shape[-2]])

            # recent 
#              self.position = tf.math.cumsum(self.real_mask, axis=1, reverse=True)
            #  self.recent_mask = tf.logical_and(self.position >= 1, self.position <= self.recent_k)
            #  self.real_recent_mask = tf.where(self.recent_mask, tf.ones_like(self.recent_mask, dtype=tf.float32), tf.zeros_like(self.recent_mask, dtype=tf.float32))
            #  self.recent_embedding_mean = tf.reduce_sum(self.seq*tf.expand_dims(self.real_recent_mask, -1), 1)/tf.reduce_sum(self.real_recent_mask, 1, keepdims=True)
#              #  self.recent_embedding_sum = tf.reduce_sum(self.seq*tf.expand_dims(self.real_recent_mask, -1), 1)

            # gru
            #  _, final_state = dynamic_rnn(
                #  GRUCell(self.hidden_size),
                #  inputs=self.seq,
                #  sequence_length=self.sequence_length,
                #  dtype=tf.float32,
                #  scope="gru",
            #  )

            # augru
            #  _, alphas = self._attention_fcn(self.target_item_embedding, self.seq, 'AGRU', False, return_alpha=True)
            #  _, final_state = dynamic_rnn_dien(
                #  VecAttGRUCell(self.hparams.hidden_size),
                #  inputs=self.seq,
                #  att_scores = tf.expand_dims(alphas, -1),
                #  sequence_length=self.sequence_length,
                #  dtype=tf.float32,
                #  scope="gru"
            #  )

            ## after atte
            #  attention_output, alphas = self._attention_fcn(self.target_item_embedding, self.seq, 'Att', False, return_alpha=True)
            #  att_fea = tf.reduce_sum(attention_output, 1)

            #  tf.summary.histogram('att_fea', att_fea)
        # MLP
        #  model_output = tf.concat([self.target_item_embedding, self.hist_embedding_sum], -1)
        #  model_output = tf.concat([self.target_item_embedding, self.hist_embedding_mean], -1)
        #  model_output = tf.concat([self.target_item_embedding, self.recent_embedding_mean], -1)
        #  model_output = tf.concat([self.target_item_embedding, self.recent_embedding_sum], -1)
        #  model_output = tf.concat([self.target_item_embedding, final_state], -1)
        #  model_output = tf.concat([self.target_item_embedding, final_state, att_fea], -1)
        #  model_output = tf.concat([self.target_item_embedding, att_fea], -1)
        model_output = tf.concat([self.target_item_embedding, self.hist_embedding_mean], -1)
        #  import pdb; pdb.set_trace()
        #  model_output = tf.concat([self.target_item_embedding, self.hist_embedding_concat], -1)
        #  model_output = self.hist_embedding_concat
        # Inner Product
        #  model_output = tf.reduce_sum(self.target_item_embedding * self.recent_embedding_mean, axis=-1)
        tf.summary.histogram("model_output", model_output)
        return model_output, attn_score_relation

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
                            with_qk=False):
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
            
            # Linear projections
            # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
            # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            Q = tf.layers.dense(self.user_embedding, num_units, activation=None) # (N, T_q, C)
            Q_min = tf.layers.dense(self.user_embedding, num_units, activation=None) # (N, T_q, C)

            K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            K_min = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)

            V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            
            # Split and concat
            # Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
            Q_ = tf.concat(tf.split(Q, num_heads, axis=1), axis=0)
            Q_min_ = tf.concat(tf.split(Q_min, num_heads, axis=1), axis=0)

            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
            K_min_ = tf.concat(tf.split(K_min, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

            # print(K_.shape)
            # print(Q_.shape)
            outputs = tf.matmul(K_, tf.expand_dims(Q_, -1))
            outputs_min = tf.matmul(K_min_, tf.expand_dims(Q_min_, -1))

            # print(outputs.shape)
            # import pdb
            # pdb.set_trace()
            outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)
            outputs_min = outputs_min / (K.get_shape().as_list()[-1] ** 0.5)
            bpr_score = tf.tile(outputs, [1, 1, K.get_shape().as_list()[1]])
            bpr_score_min = tf.transpose(tf.tile(outputs_min, [1, 1, K.get_shape().as_list()[1]]), [0, 2, 1])
            # K_ = tf.transpose(Q_, [0, 2, 1])
            outputs = bpr_score - bpr_score_min # (h*N, T_q, T_k, 1)
            # Multiplication
            # outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
            
            # Scale
            # outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            
            # Key Masking
            # tf.sign输出-1,0,1
            # 根据绝对值之和的符号判定是否mask，效果：某个sequence的特征全为0时（之前被mask过了），mask值为0，否则为1
            key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
            
            # 和下面query mask的区别：mask值不是设为0，而是设置为无穷小负值（原因是下一步要进行softmax，如果if不执行）
            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
      
            # Causality = Future blinding
            if causality:
                # 构建下三角为1的tensor
                diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
       
                paddings = tf.ones_like(masks)*(-2**32+1)
                # 下三角置为无穷小负值（原因是下一步要进行softmax）
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
      
            # Activation
            outputs = tf.nn.tanh(outputs) #tanh?
            # outputs = tf.nn.softmax(outputs * tf.abs(self.relation_mask)) * tf.sign(self.relation_mask)            
            # outputs = (2 * tf.nn.sigmoid(outputs) * tf.math.sqrt(tf.math.sqrt(tf.abs(self.relation_mask)))) / (tf.expand_dims(tf.reduce_sum(self.real_mask, 1, keep_dims=True),1))
            # / self.sequence_length # (h*N, T_q, T_k)
             
            # Query Masking
            # tf.sign输出-1,0,1
            # 根据绝对值之和的符号判定是否mask，效果：某个sequence的特征全为0时（之前被mask过了），mask值为0，否则为1
            query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1)) # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
            outputs_relation = outputs * query_masks # broadcasting. (N, T_q, C)
            # outputs_relation = outputs * 
            # tf.summary.histogram('attn_score', outputs_relation)

            # Dropouts
            #  outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
                   
            # Weighted sum
            outputs = (tf.matmul(outputs_relation, V_)) # ( h*N, T_q, C/h)
            
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
                  
            # Residual connection
            outputs += queries
                  
            # Normalize
            #outputs = normalize(outputs) # (N, T_q, C)
     
        if with_qk: return Q,K
        else: return outputs, outputs_relation



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
                return output, att_weights


