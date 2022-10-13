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
from tensorflow.python.keras.engine.base_layer import Layer
from math import sqrt
import math
import numpy as np
import socket
from reco_utils.recommender.deeprec.models.sequential import fast_attention

__all__ = ["PerformerModel"]


class PerformerModel(SLI_RECModel):

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
        """The main function to create performer model.
        
        Returns:
            obj:the output of performer section.
        """
        with tf.name_scope('performer'):
            self.seq = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            self.seq = self.seq + self.position_embedding
            self.mask = self.iterator.mask
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.sequence_length = tf.reduce_sum(self.mask, 1)
            #  attention_output = self._attention_fcn(self.target_item_embedding, hist_input)
            #  att_fea = tf.reduce_sum(attention_output, 1)
            # hyper-parameters
            self.dropout_rate = 0.0
            self.num_blocks = 2
            #  self.num_blocks = 1
            #  self.num_blocks = 3
            #  self.num_blocks = 5
            self.hidden_units = self.item_embedding_dim + self.cate_embedding_dim
            self.num_heads = 1 # prob-atte中的squeeze会把head=1压没，后面改一下squeeze
            self.is_training = True
            #  self.recent_k = 5
            self.recent_k = 1

            # prob-atte hyper-para
            #  self.factor = 10
            #  self.factor = 7
            self.factor = 5
            #  self.factor = 3
            #  self.factor = 1
            self.scale = None
            #  self.mask_flag = True
            self.mask_flag = False
            #  self.dropout = tf.keras.layers.Dropout(0.1)
            self.dropout = tf.keras.layers.Dropout(0.0)

            n_heads = self.num_heads
            d_model = self.hidden_units
            d_keys = self.hidden_units
            d_values = self.hidden_units
            num_heads = self.num_heads
            #  d_keys = d_keys or (d_model//n_heads)
            d_keys = d_model//n_heads
            #  d_values = d_values or (d_model//n_heads)
            d_values = d_model//n_heads
            self.d_model = d_model

            self.query_projection = tf.keras.layers.Dense(d_keys * n_heads)
            self.key_projection = tf.keras.layers.Dense(d_keys * n_heads)
            self.value_projection = tf.keras.layers.Dense(d_values * n_heads)
            self.out_projection = tf.keras.layers.Dense(d_model)
            self.n_heads = self.num_heads

            # Conv
            self.downConv = tf.keras.layers.Conv1D(
                          filters=self.hidden_units,
                          kernel_size=3,
                          padding='causal')
            self.norm = tf.keras.layers.BatchNormalization()
            self.activation = tf.keras.layers.ELU()
            self.maxPool = tf.keras.layers.MaxPool1D(pool_size=3, strides=2)

            # previous atte
            attention_output, alphas = self._attention_fcn(self.target_item_embedding, self.seq, 'Att', False, return_alpha=True)
            att_fea = tf.reduce_sum(attention_output, 1)

            # concat his and target
            #  self.seq = tf.concat(
                #  [self.seq, tf.expand_dims(self.target_item_embedding, 1)], 1
            #  )
            #  self.mask = tf.concat(
                #  [self.mask, tf.ones([tf.shape(self.seq)[0], 1], tf.int32)], -1
            #  )
            #  self.real_mask = tf.cast(self.mask, tf.float32)

            # Dropout
            #  self.seq = tf.layers.dropout(self.seq,
                                         #  rate=self.dropout_rate,
                                         #  training=tf.convert_to_tensor(self.is_training))
            self.seq *= tf.expand_dims(self.real_mask, -1)

            # Build blocks
            # 堆叠的注意力，参数为2
            #  att_tmp = []
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # din
                    #  attention_output, alphas = self._attention_fcn(self.target_item_embedding, self.seq, 'Att', False, return_alpha=True)
                    #  att_fea = tf.reduce_sum(attention_output, 1)
                    #  att_tmp.append(att_fea)

                    # Self-attention
#                      self.seq = self.multihead_attention(queries=self.normalize(self.seq),
                                                   #  keys=self.seq,
                                                   #  num_units=self.hidden_units,
                                                   #  num_heads=self.num_heads,
                                                   #  dropout_rate=self.dropout_rate,
                                                   #  is_training=self.is_training,
                                                   #  causality=True,
                                                   #  #  causality=False,
#                                                     scope="self_attention")
                    # prob-attention
                    #  self.seq = self.prob_attention([self.seq, self.seq, self.seq])
                    # 加了norm后，效果提升一个点
                    #  self.seq = self.prob_attention([self.normalize(self.seq), self.seq, self.seq])
                    #  下面有norm，不必再加
                    #  self.seq = self.normalize(self.seq)

                    ## performer
                    #  query = self.seq
                    #  key = self.seq
                    #  value = self.seq
                    #  softmax激活的变换kernal
                    #  kernel_transformation = fast_attention.softmax_kernel_transformation
                    # 还需要投影矩阵
                    #  num_random_features = 350
                    #  dim = 4
#                      projection_matrix = fast_attention.create_projection_matrix(num_random_features, dim)
                    #  # 将casual传入为false
                    #  attention_block_output = fast_attention.favor_attention(self.normalize(query), key, value, kernel_transformation, False, projection_matrix)
                    #  #  attention_block_output = fast_attention.favor_attention(query, key, value, kernel_transformation, False, projection_matrix)
#                      self.seq = attention_block_output

                    hidden_size = 40
                    num_heads = 1
                    #  num_heads = 4
                    #  num_heads = 8
                    dropout = 0.0
                    layer = fast_attention.SelfAttention(hidden_size, num_heads, dropout)
                    bias = tf.ones([1])
                    # 1.none cache
                    cache = None
                    # 2. cache (bad performance)
#                      dim_per_head = hidden_size // num_heads
                    #  cache = {
                    #      #  "k": tf.zeros([1, 0, num_heads, dim_per_head]),
                    #      "k": tf.zeros([tf.shape(self.seq)[0], tf.shape(self.seq)[1], num_heads, dim_per_head]),
                    #      #  "v": tf.zeros([1, 0, num_heads, dim_per_head]),
                    #      "v": tf.zeros([tf.shape(self.seq)[0], tf.shape(self.seq)[1], num_heads, dim_per_head]),
#                      }

                    self.seq = layer(self.seq, bias, training=True, cache=cache)
                    # norm all
                    #  self.seq = layer(self.normalize(self.seq), bias, training=True, cache=cache)
                    # only norm query
                    #  self.seq = layer(self.normalize(self.seq), self.seq, bias, training=True, cache=cache)

                    # Feed forward
                    self.seq = self.feedforward(self.normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate, is_training=self.is_training)
                    self.seq *= tf.expand_dims(self.real_mask, -1)

                    #  self.seq = self.normalize(self.seq)

                    #  self.hist_embedding_mean = tf.reduce_sum(self.seq*tf.expand_dims(self.real_mask, -1), 1)/tf.reduce_sum(self.real_mask, 1, keepdims=True)
                    #  att_tmp.append(self.hist_embedding_mean)

            self.seq = self.normalize(self.seq)

            #  self.seq = self.downConv(self.seq)
            #  self.seq = self.norm(self.seq)
            #  self.seq = self.activation(self.seq)
            #  self.seq = self.maxPool(self.seq)

            # all 
            #  self.hist_embedding_sum = tf.reduce_sum(self.seq*tf.expand_dims(self.real_mask, -1), 1)
            self.hist_embedding_mean = tf.reduce_sum(self.seq*tf.expand_dims(self.real_mask, -1), 1)/tf.reduce_sum(self.real_mask, 1, keepdims=True)
            #  self.hist_embedding_mean = tf.reduce_mean(self.seq, 1)
            #  self.hist_embedding_mean = tf.reduce_max(self.seq, 1)

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
        #  model_output = tf.concat([self.target_item_embedding, self.hist_embedding_mean, self.recent_embedding_mean], -1)
        #  model_output = tf.concat([self.target_item_embedding, final_state], -1)
        #  model_output = tf.concat([self.target_item_embedding, att_fea], -1)
        model_output = tf.concat([self.target_item_embedding, self.hist_embedding_mean, att_fea], -1)
        # Inner Product
        #  model_output = tf.reduce_sum(self.target_item_embedding * self.recent_embedding_mean, axis=-1)

        #  att_tmp.append(self.target_item_embedding)
        #  model_output = tf.concat(att_tmp, -1)
        tf.summary.histogram("model_output", model_output)
        return model_output

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
            Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            
            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

            # Multiplication
            #  outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
            B, L, D = queries.shape.as_list()
            _, S, _ = keys.shape.as_list()
            U = self.factor * np.ceil(np.log(S)).astype('int').item()
            u = self.factor * np.ceil(np.log(L)).astype('int').item()
            #  import pdb; pdb.set_trace()
            # (?, 20, 50)，(?, 20)
            scores_top, index = self._prob_QK_new(Q_, K_, u, U)
            outputs = scores_top
            batch_indexes = tf.tile(tf.range(tf.shape(Q)[0])[:, tf.newaxis], (1, tf.shape(index)[-1]))
            idx = tf.stack(values=[batch_indexes, index], axis=-1)
            queries_tmp = tf.gather_nd(queries, idx)
            
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            
            # Key Masking
            # tf.sign输出-1,0,1
            # 根据绝对值之和的符号判定是否mask，效果：某个sequence的特征全为0时（之前被mask过了），mask值为0，否则为1
            key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries_tmp)[1], 1]) # (h*N, T_q, T_k)
            
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
            outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
             
            # Query Masking
            # tf.sign输出-1,0,1
            # 根据绝对值之和的符号判定是否mask，效果：某个sequence的特征全为0时（之前被mask过了），mask值为0，否则为1
            query_masks = tf.sign(tf.reduce_sum(tf.abs(queries_tmp), axis=-1)) # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
            outputs *= query_masks # broadcasting. (N, T_q, C)
              
            # Dropouts
            #  outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
                   
            # Weighted sum
            #  outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
            batch_indexes = tf.tile(tf.range(tf.shape(V)[0])[:, tf.newaxis], (1, tf.shape(index)[-1]))
            idx = tf.stack(values=[batch_indexes, index], axis=-1)
            outputs = tf.tensor_scatter_nd_update(V_, idx, tf.matmul(outputs, V_))
            
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
                  
            # Residual connection
            outputs += queries
                  
            # Normalize
            #outputs = normalize(outputs) # (N, T_q, C)
     
        if with_qk: return Q,K
        else: return outputs

    def prob_attention(self, inputs, attn_mask=None):
        """
        线性变换，输入根据head变形，传入innter_attention，输出变形回来
        """

        queries, keys, values = inputs
        #  B, L, _ = queries.shape
        B, L, D = queries.shape.as_list()
        #  B, L, _ = tf.shape(queries)[0], tf.shape(queries)[1], tf.shape(queries)[2]
        #  B, L, _ = queries.get_shape()
        #  _, S, _ = keys.shape
        _, S, _ = keys.shape.as_list()
        #  _, S, _ = tf.shape(keys)[0],tf.shape(keys)[1],tf.shape(keys)[2],
        #  _, S, _ = keys.get_shape()
        H = self.n_heads

        #  self.queries = tf.reshape(self.query_projection(queries), (B, L, H, -1))
        #  self.keys = tf.reshape(self.key_projection(keys), (B, S, H, -1))
        #  self.values = tf.reshape(self.value_projection(values), (B, S, H, -1))
        self.queries = tf.reshape(self.query_projection(queries), (-1, L, H, D//H))
        self.keys = tf.reshape(self.key_projection(keys), (-1, S, H, D//H))
        self.values = tf.reshape(self.value_projection(values), (-1, S, H, D//H))
        #  self.queries = tf.reshape(tf.layers.dense(queries, D, activation=None), (-1, L, H, D//H))
        #  self.keys = tf.reshape(tf.layers.dense(keys, D, activation=None), (-1, S, H, D//H))
        #  self.values = tf.reshape(tf.layers.dense(values, D, activation=None), (-1, S, H, D//H))
#
        #  out = tf.reshape(self.inner_attention([self.queries, self.keys, self.values], attn_mask=attn_mask), (B, L, -1))
        out = tf.reshape(self.inner_attention([self.queries, self.keys, self.values], attn_mask=attn_mask), (-1, L, D))

        #  return self.out_projection(out)
        return out
        #  return out + queries

    def inner_attention(self, inputs, attn_mask=None):
        queries, keys, values = inputs
        #  B, L, H, D = queries.shape
        #  _, S, _, _ = keys.shape
        B, L, H, D = queries.shape.as_list()
        _, S, _, _ = keys.shape.as_list()
        #  B, L, H, D = queries.get_shape()
        #  _, S, _, _ = keys.get_shape()

        """
        这里的reshape的作用是？如果head为1的话，结果并不影响
        """
        #  queries = tf.reshape(queries, (B, H, L, -1))
        #  keys = tf.reshape(keys, (B, H, S, -1))
        #  values = tf.reshape(values, (B, H, S, -1))
        queries = tf.reshape(queries, (-1, H, L, D))
        keys = tf.reshape(keys, (-1, H, S, D))
        values = tf.reshape(values, (-1, H, S, D))

        U = self.factor * np.ceil(np.log(S)).astype('int').item()
        u = self.factor * np.ceil(np.log(L)).astype('int').item()

        scores_top, index = self._prob_QK(queries, keys, u, U)
        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # 应该用初始values，而非线性变换后的
        # get the context
        context = self._get_initial_context(values, L)
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index, L)

        return context

    def _prob_QK_new(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        #  B, H, L, E = K.shape
        #  _, _, S, _ = Q.shape
        B, L, E = K.shape.as_list()
        _, S, _ = Q.shape.as_list()

        # calculate the sampled Q_K
        #  K_expand = tf.broadcast_to(tf.expand_dims(K, -3), (B, H, S, L, E))
        K_expand = tf.broadcast_to(tf.expand_dims(K, -3), (tf.shape(K)[0], S, L, E))
        #  K_expand = tf.tile(tf.expand_dims(K, -3), (1, 1, S, 1, 1))

        # 最大值为L，长度分别为S和sample_k的均匀分布
        indx_q_seq = tf.random.uniform((S,), maxval=L, dtype=tf.int32)
        indx_k_seq = tf.random.uniform((sample_k,), maxval=L, dtype=tf.int32)

        # 这行好像没有用吧
        K_sample = tf.gather(K_expand, tf.range(S), axis=1)

        # 这行好像没有用吧
        K_sample = tf.gather(K_sample, indx_q_seq, axis=1)
        K_sample = tf.gather(K_sample, indx_k_seq, axis=2)

        # 计算attention分数
        # B,S,1,E * B,S,E,L -> B,H,S,1,L  -> B,H,S,L
        #  Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q, -2), tf.einsum("...ij->...ji", K_sample)))
        Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q, -2), tf.einsum("...ij->...ji", K_sample)), -2)
        # find the Top_k query with sparisty measurement
        M = tf.math.reduce_max(Q_K_sample, axis=-1) - tf.raw_ops.Div(x=tf.reduce_sum(Q_K_sample, axis=-1), y=L)
        #  import pdb; pdb.set_trace()
        M_top = tf.math.top_k(M, n_top, sorted=False)[1]
        #  不缩减呢
        #  M_top = tf.argsort(M, direction='DESCENDING', stable=True, axis=-1) # B*L -> B*L
        #  batch_indexes = tf.tile(tf.range(Q.shape[0])[:, tf.newaxis, tf.newaxis], (1, Q.shape[1], n_top))
        batch_indexes = tf.tile(tf.range(tf.shape(Q)[0])[:, tf.newaxis], (1, n_top))
        #  head_indexes = tf.tile(tf.range(Q.shape[1])[tf.newaxis, :, tf.newaxis], (Q.shape[0], 1, n_top))

        #  import pdb; pdb.set_trace()
        idx = tf.stack(values=[batch_indexes, M_top], axis=-1)

        # use the reduced Q to calculate Q_K
        Q_reduce = tf.gather_nd(Q, idx)

        Q_K = tf.matmul(Q_reduce, tf.transpose(K, [0, 2, 1]))

        return Q_K, M_top

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        #  B, H, L, E = K.shape
        #  _, _, S, _ = Q.shape
        B, H, L, E = K.shape.as_list()
        _, _, S, _ = Q.shape.as_list()

        # calculate the sampled Q_K
        #  K_expand = tf.broadcast_to(tf.expand_dims(K, -3), (B, H, S, L, E))
        K_expand = tf.broadcast_to(tf.expand_dims(K, -3), (tf.shape(K)[0], H, S, L, E))
        #  K_expand = tf.tile(tf.expand_dims(K, -3), (1, 1, S, 1, 1))

        # 最大值为L，长度分别为S和sample_k的均匀分布
        indx_q_seq = tf.random.uniform((S,), maxval=L, dtype=tf.int32)
        indx_k_seq = tf.random.uniform((sample_k,), maxval=L, dtype=tf.int32)

        # 这行好像没有用吧
        K_sample = tf.gather(K_expand, tf.range(S), axis=2)

        # 这行好像没有用吧
        K_sample = tf.gather(K_sample, indx_q_seq, axis=2)
        K_sample = tf.gather(K_sample, indx_k_seq, axis=3)

        # 计算attention分数
        # B,H,S,1,E * B,H,S,E,L -> B,H,S,1,L  -> B,H,S,L
        #  Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q, -2), tf.einsum("...ij->...ji", K_sample)))
        if 'kwai' in socket.gethostname():
            Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q, -2), tf.transpose(K_sample, [0,1,2,4,3])), -2)
        else:
            Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q, -2), tf.einsum("...ij->...ji", K_sample)), -2)
        # find the Top_k query with sparisty measurement
        if 'kwai' in socket.gethostname():
            M = tf.math.reduce_max(Q_K_sample, axis=-1) - tf.raw_ops.Div(x=tf.reduce_sum(Q_K_sample, axis=-1), y=L)
        else:
            M = tf.math.reduce_max(Q_K_sample, axis=-1) - tf.div(x=tf.reduce_sum(Q_K_sample, axis=-1), y=L)
        #  import pdb; pdb.set_trace()
        M_top = tf.math.top_k(M, n_top, sorted=False)[1]
        #  batch_indexes = tf.tile(tf.range(Q.shape[0])[:, tf.newaxis, tf.newaxis], (1, Q.shape[1], n_top))
        batch_indexes = tf.tile(tf.range(tf.shape(Q)[0])[:, tf.newaxis, tf.newaxis], (1, tf.shape(Q)[1], n_top))
        #  head_indexes = tf.tile(tf.range(Q.shape[1])[tf.newaxis, :, tf.newaxis], (Q.shape[0], 1, n_top))
        head_indexes = tf.tile(tf.range(tf.shape(Q)[1])[tf.newaxis, :, tf.newaxis], (tf.shape(Q)[0], 1, n_top))

        #  import pdb; pdb.set_trace()
        idx = tf.stack(values=[batch_indexes, head_indexes, M_top], axis=-1)

        # use the reduced Q to calculate Q_K
        Q_reduce = tf.gather_nd(Q, idx)

        Q_K = tf.matmul(Q_reduce, tf.transpose(K, [0, 1, 3, 2]))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        为啥要把value在embedding维累加
        """
        #  B, H, L_V, D = V.shape
        B, H, L_V, D = V.shape.as_list()
        if not self.mask_flag:
            V_sum = tf.reduce_sum(V, -2)
            #  import pdb; pdb.set_trace()
            #  contex = tf.identity(tf.broadcast_to(tf.expand_dims(V_sum, -2), [B, H, L_Q, V_sum.shape[-1]]))
            #  contex = tf.identity(tf.tile(tf.expand_dims(V_sum, -2), [1, 1, L_Q, 1]))
            #  contex = tf.identity(tf.broadcast_to(tf.expand_dims(V_sum, -2), [B, H, L_Q, V_sum.shape[-1]]))
            contex = tf.identity(tf.broadcast_to(tf.expand_dims(V_sum, -2), [tf.shape(V)[0], H, L_Q, V_sum.shape[-1]]))
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = tf.math.cumsum(V, axis=-1)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        #  B, H, L_V, D = V.shape
        B, H, L_V, D = V.shape.as_list()
        B = tf.shape(V)[0]

        if self.mask_flag:
            #  attn_mask = ProbMask(B, H, L_Q, index, scores)
            attn_mask = ProbMask(B, H, L_Q, index, scores)

            # scores.masked_fill_(attn_mask.mask, -np.inf)
            num = 3.4 * math.pow(10, 38)
            scores = (scores * attn_mask.mask) + (-((attn_mask.mask * num + num) - num))

        attn = tf.keras.activations.softmax(scores, axis=-1)  # nn.Softmax(dim=-1)(scores)
        #  batch_indexes = tf.tile(tf.range(V.shape[0])[:, tf.newaxis, tf.newaxis], (1, V.shape[1], index.shape[-1]))
        #  head_indexes = tf.tile(tf.range(V.shape[1])[tf.newaxis, :, tf.newaxis], (V.shape[0], 1, index.shape[-1]))
        batch_indexes = tf.tile(tf.range(tf.shape(V)[0])[:, tf.newaxis, tf.newaxis], (1, tf.shape(V)[1], tf.shape(index)[-1]))
        head_indexes = tf.tile(tf.range(tf.shape(V)[1])[tf.newaxis, :, tf.newaxis], (tf.shape(V)[0], 1, tf.shape(index)[-1]))

        idx = tf.stack(values=[batch_indexes, head_indexes, index], axis=-1)

        # Issue: tensor update 05.28 21:00
        #  context_in = tf.tensor_scatter_nd_update(context_in, idx, tf.matmul(attn, V))
        #  context_in: (?, 1, 250, 40); idx: (?, 1, 30, 3);  tf.matmul(attn, V): (?, 1, 30, 250) * (?, 1, 250, 40)
        # Issue: tensor update 05.28 21:00
        replacement = tf.matmul(attn, V)
        update = tf.scatter_nd(idx, replacement, tf.shape(context_in))
        mask = tf.scatter_nd(idx, tf.ones_like(replacement, dtype=tf.bool), tf.shape(context_in))
        context_in = tf.where(mask, update, context_in)

        return tf.convert_to_tensor(context_in)


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


class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]

        mask_a = tf.linalg.band_part(tf.ones(mask_shape), 0, -1)  # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(tf.ones(mask_shape), 0, 0)  # Diagonal matrix of 0s and 1s
        mask = tf.cast(mask_a - mask_b, dtype=tf.float32)

        self._mask = mask
        tf.stop_gradient(self._mask)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores):
        _mask = tf.ones((L, scores.shape[-1]))

        mask_a = tf.linalg.band_part(_mask, 0, -1)  # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(_mask, 0, 0)  # Diagonal matrix of 0s and 1s
        _mask = tf.cast(mask_a - mask_b, dtype=tf.float32)

        #  import pdb; pdb.set_trace()
        _mask_ex = tf.broadcast_to(_mask, [B, H, L, scores.shape[-1]])
        #  _mask_ex = tf.tile(tf.expand_dims(tf.expand_dims(_mask, 0), 0), [B, H, 1, 1])
        #  _mask_ex = tf.broadcast_to(tf.expand_dims(tf.expand_dims(_mask, 0), 0), [B, H, L, scores.shape[-1]])
        #  indicator = _mask_ex[tf.expand_dims(tf.range(B),[1,2]),
                    #  tf.expand_dims(tf.range(H),[0,2]),
                    #  index, :]
        #  indicator = _mask_ex[tf.range(B)[:, None, None], tf.range(H)[None, :, None], index, :]

        batch_indexes = tf.tile(tf.range(B)[:, tf.newaxis, tf.newaxis], (1, H, tf.shape(index)[-1]))
        head_indexes = tf.tile(tf.range(H)[tf.newaxis, :, tf.newaxis], (B, 1, tf.shape(index)[-1]))
        idx = tf.stack(values=[batch_indexes, head_indexes, index], axis=-1)
        indicator = tf.gather_nd(_mask_ex, idx)

        #  self._mask = indicator.reshape(scores.shape)
        #  self._mask = tf.reshape(indicator, [tf.shape(scores)[0]] + scores.shape.as_list()[1:])
        self._mask = tf.reshape(indicator, [-1] + scores.shape.as_list()[1:])

    @property
    def mask(self):
        return self._mask

