# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from time import time
import socket
import os
import _pickle as cPickle
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from reco_utils.recommender.deeprec.deeprec_utils import load_dict
from tensorflow.contrib.rnn import GRUCell
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import (
    VecAttGRUCell,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.models.sequential.rnn_dien import dynamic_rnn as dynamic_rnn_dien

## diffpool
from tensorflow.keras import backend as K

from reco_utils.recommender.deeprec.models.sequential.gcn_utils import (
    GCN, GAT,
    gcn, gat,
    aggregate_neighbors, sum_updater, sum_reducer, identity_updater, segment_softmax, gcn_mapper, add_self_loop_edge,
    init_gat_weight, init_gcn_weight, init_score_gcn_weight,
    topk_pool, sample_new_graph_by_node_index, mean_pool, max_pool, segment_top_k,
    repeat, maximum_aggregator
)

__all__ = ["GCNModel"]


class GCNModel(SequentialBaseModel):

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization of variables for caser

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
        """
        self.hparams = hparams
        self.train_file = hparams.train_dir
        self.path = hparams.graph_dir
        self.n_layers = 2
        self.smoothness_ratio = 0.1
        self.degree_ratio = 0.1
        self.sparsity_ratio = 0.1
        self.watch_A_S_A = False
        self.watch_length = False

        ## Hyperparameter study - threshold 
        #  设置阈值（先mask再knn），缺点是不知道阈值的分布
        #  设置统一的topk（先mask再knn），缺点是有些序列很短，导致完全图
        #  设置变化的topk（先mask再knn），缺点是
        #  self.threshold = 0.0
        #  self.threshold = 0.1
        #  self.threshold = 0.2
        #  self.threshold = 0.3 # best in att
        #  self.threshold = 0.4
        self.threshold = 0.5 # best in cos
        #  self.threshold = 0.6
        #  self.threshold = 0.7
        #  self.threshold = 0.8
        #  self.threshold = 0.9
        #  self.threshold = 1.0

        #  self.pool_ratio = 0.1
        #  self.pool_ratio = 0.2
        #  self.pool_ratio = 0.3
        #  self.pool_ratio = 0.4 # best
        #  self.pool_ratio = 0.5
        #  self.pool_ratio = 0.6
        #  self.pool_ratio = 0.7
        #  self.pool_ratio = 0.8
        #  self.pool_ratio = 0.9
        #  self.pool_ratio = 1

        #  self.pool_length = 50
        #  self.pool_length = 40
        self.pool_length = 30
        #  self.pool_length = 20
        #  self.pool_length = 10
        #  self.pool_length = 45

        #  self.pool_layers = 1
        self.pool_layers = 2 # best

        self.query_shared = True
        #  self.query_shared = False

        super().__init__(hparams, iterator_creator, seed=None)


    def _build_seq_graph(self):
        #  output = self._gru()
        #  output = self._sum()
        #  output = self._din()
        #  output = self._dien() # only predict
        #  output = self._diffpool()
        #  output = self._topkpool()
        #  output = self._sagpool()
        #  output = self._full_batch_sagpool()
        #  output = self._pool()
        output = self._SURGE()
        return output

    def _sum(self):
        with tf.name_scope('sum'):
            #  hist_input = tf.concat(
                #  [self.item_history_embedding, self.cate_history_embedding], 2
            #  )
            # add for cate
            hist_input = self.item_history_embedding
            self.target_item_embedding = self.target_item_embedding[:,:40]
            #  self.hist_embedding_sum = tf.reduce_sum(hist_input, 1)
            self.mask = self.iterator.mask
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.hist_embedding_sum = tf.reduce_sum(hist_input*tf.expand_dims(self.real_mask, -1), 1)

        model_output = tf.concat([self.target_item_embedding, self.hist_embedding_sum], -1)
        tf.summary.histogram("model_output", model_output)
        return model_output
 
    def _din(self):
        """The main function to use din prediction model.
        
        Returns:
            obj:the output of gcn section.
        """
        hparams = self.hparams
        with tf.name_scope('gcn'):
            #  hist_input = tf.concat(
                #  [self.item_history_embedding, self.cate_history_embedding], 2
            #  )
            # add for cate
            hist_input = self.item_history_embedding
            self.target_item_embedding = self.target_item_embedding[:,:40]
            self.hist_embedding_sum = tf.reduce_sum(hist_input, 1)
            self.mask = self.iterator.mask

            #  #  recent
            #  self.real_mask = self.iterator.mask
            #  self.position = tf.math.cumsum(self.real_mask, axis=1, reverse=True)
            #  #  self.recent_mask = tf.logical_and(self.position >= 1, self.position <= hparams.contrastive_recent_k)
            #  self.recent_mask = tf.logical_and(self.position >= 1, self.position <= 5)
            #  self.real_recent_mask = tf.where(self.recent_mask, tf.ones_like(self.recent_mask, dtype=tf.float32), tf.zeros_like(self.recent_mask, dtype=tf.float32))
            #  #  self.hist_recent = tf.reduce_sum(hist_input*tf.expand_dims(self.real_recent_mask, -1), 1)/tf.reduce_sum(self.real_recent_mask, 1, keepdims=True)
            #  self.mask = self.real_mask

            #  attention_output = self._attention_fcn(self.target_item_embedding, hist_input)
            attention_output = self._attention_fcn(self.target_item_embedding, hist_input, '00', False)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)

        model_output = tf.concat([self.target_item_embedding, self.hist_embedding_sum, att_fea], -1)
        tf.summary.histogram("model_output", model_output)
        return model_output
    
    def _last_or_recent(self):  
        """The main function to use last/recent items to predict.
        
        Returns:
            obj:the output of gcn section.
        """
        hparams = self.hparams
        with tf.name_scope('gcn'):

            #  hist_input = tf.concat(
                #  [self.item_history_embedding, self.cate_history_embedding], 2
            #  )
            hist_input = self.item_history_embedding
            self.target_item_embedding = self.target_item_embedding[:,:40]

            # last item
            #  #  self.last_item_embedding = self.item_history_embedding[:,-1,:] # wrong
            #  self.mask = self.iterator.mask
            #  self.sequence_length = tf.reduce_sum(self.mask, 1)
            #  self.last_item_embedding = tf.gather_nd(hist_input, tf.expand_dims(self.sequence_length - 1, -1), batch_dims=1)

            # recent k items
            self.real_mask = self.iterator.mask
            self.position = tf.math.cumsum(self.real_mask, axis=1, reverse=True)
            #  self.recent_mask = tf.logical_and(self.position >= 1, self.position <= hparams.contrastive_recent_k)
            self.recent_mask = tf.logical_and(self.position >= 1, self.position <= 5)
            self.real_recent_mask = tf.where(self.recent_mask, tf.ones_like(self.recent_mask, dtype=tf.float32), tf.zeros_like(self.recent_mask, dtype=tf.float32))
            self.hist_recent = tf.reduce_sum(hist_input*tf.expand_dims(self.real_recent_mask, -1), 1)/tf.reduce_sum(self.real_recent_mask, 1, keepdims=True)

            # >>> use attention >>>
            concat_all = tf.concat(
                [
                    self.user_embedding,
                    self.target_item_embedding,
                    #  self.last_item_embedding,
                    self.hist_recent,
                    #  tf.expand_dims(self.iterator.time_to_now[:, -1], -1),
                ],
                1,
            )
            last_hidden_nn_layer = concat_all
            alpha_logit = self._fcn_net(
                last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="fcn_alpha"
            )
            alpha_output = tf.sigmoid(alpha_logit)
            tf.summary.histogram("alpha", alpha_output)

            #  long_short_embed = self.user_embedding * alpha_output + self.last_item_embedding * (1.0 - alpha_output)
            long_short_embed = self.user_embedding * alpha_output + self.hist_recent * (1.0 - alpha_output)
            # <<< use attention <<<

        # last
        #  model_output = tf.concat([self.user_embedding, self.last_item_embedding, self.target_item_embedding], -1)
        # recent
        #  model_output = tf.concat([self.user_embedding, self.hist_recent, self.target_item_embedding], -1)
        # attetion
        model_output = tf.concat([long_short_embed, self.target_item_embedding], 1)
        tf.summary.histogram("model_output", model_output)
        return model_output

    ## dien
    def _dien(self):
        """The main function to create din model.
        
        Returns:
            obj:the output of din section.
        """
        hparams = self.hparams
        with tf.name_scope('dien'):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            # add for cate
            #  hist_input = self.item_history_embedding
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(self.mask, 1)

            #  hist_input = self.item_history_embedding[:,-5:,:]
            #  hist_recent = tf.gather_nd(hist_input, tf.expand_dims(self.sequence_length - 1, -1), batch_dims=1)
            #  hist_recent = tf.gather_nd(hist_input, tf.range(self.sequence_length - 5, self.sequence_length), batch_dims=1)
            #  self.recent_mask = tf.greater(tf.range(0, ),sequence_length)

            # recent(wrong)
            #  self.position = tf.tile(tf.reshape(tf.range(hparams.max_seq_length), [1,-1]), [hist_input.shape[0], 1])
            #  self.recent_mask = tf.logical_and(self.position >= tf.expand_dims(self.sequence_length - hparams.contrastive_recent_k, -1), self.position < tf.expand_dims(self.sequence_length, -1))
            #  self.real_recent_mask = tf.where(self.recent_mask, tf.ones_like(self.recent_mask, dtype=tf.float32), tf.zeros_like(self.recent_mask, dtype=tf.float32))
            #  self.hist_recent = tf.reduce_sum(hist_input*tf.expand_dims(self.real_recent_mask, -1), 1)/tf.reduce_sum(self.real_recent_mask, 1, keepdims=True)

            # new recent
            #  self.position = tf.math.cumsum(self.real_mask, axis=1, reverse=True)
            #  self.recent_mask = tf.logical_and(self.position >= 1, self.position <= hparams.contrastive_recent_k)
            #  self.real_recent_mask = tf.where(self.recent_mask, tf.ones_like(self.recent_mask, dtype=tf.float32), tf.zeros_like(self.recent_mask, dtype=tf.float32))
            #  self.hist_recent = tf.reduce_sum(hist_input*tf.expand_dims(self.real_recent_mask, -1), 1)/tf.reduce_sum(self.real_recent_mask, 1, keepdims=True)

            self.target_item_embedding = self.target_item_embedding[:,:40]
            self.hist_embedding_sum = tf.reduce_sum(hist_input, 1)
#              with tf.name_scope('rnn_1'):
                #  rnn_outputs, _ = dynamic_rnn_dien(
                #      GRUCell(hparams.hidden_size),
                #      inputs=hist_input,
                #      #  inputs=hist_recent,
                #      sequence_length=self.sequence_length,
                #      dtype=tf.float32,
                #      scope="gru1"
                #  )
#                  tf.summary.histogram('GRU_outputs', rnn_outputs)

            # Attention layer
            with tf.name_scope('Attention_layer_1'):
                #  _, alphas = self._attention_fcn(self.target_item_embedding, rnn_outputs, return_alpha=True)
                #  _, alphas = self._attention_fcn(self.target_item_embedding, hist_input, return_alpha=True)
                #  _, alphas = self._attention_fcn(self.target_item_embedding, hist_input, '1',return_alpha=True)
                attention_output, alphas = self._attention_fcn(self.target_item_embedding, hist_input, '1', False, return_alpha=True)
                att_fea = tf.reduce_sum(attention_output, 1)

            with tf.name_scope('rnn_2'):
                _, final_state = dynamic_rnn_dien(
                    VecAttGRUCell(hparams.hidden_size),
                    #  inputs=rnn_outputs,
                    inputs=hist_input,
                    att_scores = tf.expand_dims(alphas, -1),
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="gru2"
                )
                tf.summary.histogram('GRU2_Final_State', final_state)

        #  model_output = tf.concat([self.target_item_embedding, final_state, self.hist_embedding_sum, self.target_item_embedding*self.hist_embedding_sum], 1)
        model_output = tf.concat([self.target_item_embedding, final_state, self.hist_embedding_sum, self.target_item_embedding*self.hist_embedding_sum, att_fea], 1)
        #  model_output = tf.concat([self.target_item_embedding, final_state, self.hist_embedding_sum], 1)
        #  model_output = tf.concat([self.target_item_embedding, final_state], 1)
        tf.summary.histogram("model_output", model_output)
        return model_output


    def _gru(self):
        """The main function to create GRU4Rec model.
        
        Returns:
            obj:the output of GRU4Rec section.
        """
        hparams = self.hparams
        with tf.variable_scope("gru4rec"):
            #  self.history_embedding = tf.concat(
                #  [self.item_history_embedding, self.cate_history_embedding], 2
            #  )
            # add for cate
            hist_input = self.item_history_embedding
            self.target_item_embedding = self.target_item_embedding[:,:40]

            self.hist_embedding_sum = tf.reduce_sum(hist_input, 1)
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(self.mask, 1)

            with tf.name_scope("gru"):
                rnn_outputs, final_state = dynamic_rnn(
                    GRUCell(self.hidden_size),
                    inputs=hist_input,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="gru",
                )
                tf.summary.histogram("GRU_outputs", rnn_outputs)

            # >>> not use attention >>>
            #  model_output = tf.concat([final_state, self.target_item_embedding], 1)
            #  #  model_output = tf.concat([final_state, self.user_embedding, self.target_item_embedding], 1)
            #  tf.summary.histogram("model_output", model_output)
            #  return model_output

            # >>> use attention >>>
            concat_all = tf.concat(
                [
                    #  self.last_item_embedding,
                    final_state,
                    self.user_embedding,
                    self.target_item_embedding,
                    #  tf.expand_dims(self.iterator.time_to_now[:, -1], -1),
                ],
                1,
            )
            last_hidden_nn_layer = concat_all
            alpha_logit = self._fcn_net(
                last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="fcn_alpha"
            )
            alpha_output = tf.sigmoid(alpha_logit)
            tf.summary.histogram("alpha", alpha_output)

            # 1.mlp 
            long_short_embed = self.user_embedding * alpha_output + final_state * (1.0 - alpha_output)
            model_output = tf.concat([long_short_embed, self.target_item_embedding], 1)
            tf.summary.histogram("model_output", model_output)
            return model_output

            # 2.inner-product 
            #  logit = tf.reduce_sum(self.user_embedding*self.target_item_embedding, -1, keepdims=True) * alpha_output \
                #  + tf.reduce_sum(self.last_item_embedding*self.target_item_embedding, -1, keepdims=True) * (1.0 - alpha_output) \
            #  return logit

            # <<< use attention <<<
        
    #  def _attention_fcn(self, query, user_embedding, return_alpha=False):
    #  def _attention_fcn(self, query, user_embedding, var_id,return_alpha=False):
    def _attention_fcn(self, query, user_embedding, var_id, reuse,return_alpha=False):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item which is regarded as a query in attention operations.
            user_embedding (obj): The output of RNN layers which is regarded as user modeling.

        Returns:
            obj: Weighted sum of user modeling.
        """
        hparams = self.hparams
        #  with tf.variable_scope("attention_fcn"):
        #  with tf.variable_scope("attention_fcn", reuse=reuse):
        #  with tf.variable_scope("attention_fcn"+var_id):
        with tf.variable_scope("attention_fcn"+str(var_id), reuse=reuse):
            query_size = query.shape[1].value
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            #  scope.reuse_variables()
            attention_mat = tf.get_variable(
                #  name="attention_mat",
                name="attention_mat"+str(var_id),
                shape=[user_embedding.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(user_embedding, attention_mat, [[2], [0]])
            #  att_inputs = tf.tensordot(user_embedding, self.attention_mat, [[2], [0]])
            #  att_inputs = tf.layers.dense(user_embedding, query_size, use_bias=False)

            queries = tf.reshape(
                #  tf.tile(query, [1, att_inputs.shape[1].value]), tf.shape(att_inputs)
                #  https://blog.csdn.net/suiyueruge1314/article/details/105843759
                tf.tile(query, [1, tf.shape(att_inputs)[1]]), tf.shape(att_inputs)
            )
            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )
            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="att_fcn"
            )
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = user_embedding * tf.expand_dims(att_weights, -1)
            if not return_alpha:
                return output
            else:
                return output, att_weights


    def _interest_fusion(self, X, A, layer, reuse):

        with tf.name_scope('CGAT'):
            ## CGAT Layer ('cluster_embedding'):

            #  当层数为两层时，所有节点的非零元素就变为了所有节点数，即完全图，所以问题出在图粗化的矩阵变换？
            #  pp1 = tf.sort(tf.cast(tf.count_nonzero(A, -1), tf.float32)[0], direction='DESCENDING', axis=-1)

            # add self-loop
            A_bool = tf.cast(tf.greater(A, 0), A.dtype)
            A_bool = A_bool * (tf.ones([self.max_n_nodes,self.max_n_nodes]) - tf.eye(self.max_n_nodes)) + tf.eye(self.max_n_nodes)

            # normalize graph
            # gat只需要对attention权重进行归一化，边权重保持为1即可，gcn才需要对邻接矩阵归一化。这里的A的归一化是为了后面的感受野的平均操作
            D = tf.reduce_sum(A_bool, axis=-1) # B*L
            D = tf.sqrt(D)[:, None] + K.epsilon() # B*1*L
            A = (A_bool / D) / tf.transpose(D, perm=(0,2,1)) # B*L*L / B*1*L / B*L*1

            # >>>>>> ablation_study wo/CGAT >>>>>>>>>>

            #  簇i感受野，一阶/二阶/三阶邻居平均，作为master query，x_q对应公式中的mi。一阶邻域包含自身，因为self-loop
            X_q = tf.matmul(A, X) # B*L*F
            X_q = tf.matmul(A, X_q)
            #  X_q = tf.matmul(A, X_q)

            # 公式6，分配矩阵S，计算簇i与节点j的隶属程度，也即attention score（X_q和X用来求S）
            if not self.query_shared:
                _, f_1 = self._attention_gat(X_q, X, 'f2_layer'+str(layer), False, return_alpha=True)
                _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f1_layer'+str(layer), False, return_alpha=True)
            if self.query_shared:
                _, f_1 = self._attention_gat(X_q, X, 'f2_shared', reuse, return_alpha=True)
                _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f1_shared', reuse, return_alpha=True)

            # i和j的隶属关系要限制在一阶邻域（而不是两两item都计算隶属度，而且这样后面embedding是根据所有item来聚合的）（fix前后效果差距明显）
            # 但A是归一化后的矩阵，原始是0/0～1，而非0/1的，所以要用A_bool。
            #  S = A * f_1 + A * tf.transpose(f_2, (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
            #  S = A_bool * f_1 + A_bool * tf.transpose(f_2, (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
            S = A_bool * tf.expand_dims(f_1,1) + A_bool * tf.transpose(tf.expand_dims(f_2,1), (0,2,1)) # B*L*1 x B*L*1 -> B*L*L

            ## >>>>>>> ablation_study token2token gat >>>>>>  用center node而非mean作为master
            # method1
            #  X_k = tf.layers.dense(X, 40)
            #  S = A * tf.matmul(X_k, tf.transpose(X_k, (0,2,1))) # B*L*L
            # method2
#              X_k = tf.layers.dense(X, 40)
            #  #  f_1 = tf.layers.dense(X_k, 1)  # B*L*F x F*1 -> B*L*1
            #  #  f_2 = tf.layers.dense(X_k, 1) # B*L*F x F*1 -> B*L*1
            #  f_1 = tf.squeeze(tf.layers.dense(X_k, 1),-1)  # B*L*F x F*1 -> B*L*1
            #  f_2 = tf.squeeze(tf.layers.dense(X_k, 1),-1) # B*L*F x F*1 -> B*L*1
            #  #  S = A_bool * f_1 + A_bool * tf.transpose(f_2, (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
            #  S = A_bool * tf.expand_dims(f_1,1) + A_bool * tf.transpose(tf.expand_dims(f_2,1), (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
            ## <<<<<<< ablation_study token2token gat <<<<<<

            ## >>>>>>> ablation_study master2token gat >>>>>>  用center node而非mean作为master
            #  X_q = tf.matmul(A, X) # B*L*F
            #  X_q = tf.layers.dense(X_q, 40)
            #  X_k = tf.layers.dense(X, 40)
            #  f_1 = tf.layers.dense(X_q, 1)  # B*L*F x F*1 -> B*L*1
            #  f_2 = tf.layers.dense(X_k, 1) # B*L*F x F*1 -> B*L*1
            #  S = A * f_1 + A * tf.transpose(f_2, (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
            ## <<<<<<< ablation_study master2token gat <<<<<<

            ##  loss
            #  S_gram = tf.matmul(S, tf.transpose(S, (0,2,1)))
            #  LP_loss = A - S_gram
            #  LP_loss = tf.norm(LP_loss, axis=(-1, -2))
            #  self.LP_loss = K.mean(LP_loss)
#
            #  entr = tf.negative(tf.reduce_sum(tf.multiply(S, K.log(S + K.epsilon())), axis=-1))
            #  entr_loss = K.mean(entr, axis=-1)
            #  self.entr_loss = K.mean(entr_loss)

            S = tf.nn.leaky_relu(S)
            # 有0元素再经过softmax，变为非0，虽然很小，且后面graph粗化时候，有mask的过程，但是mask只会mask不要的节点。原始graph中两个需要节点之间的无边状态0，会被修正为非0元素。
            #  S = tf.nn.softmax(S, axis=-1)
            #  S = S * A_bool
            # softmax和bool改为如下形式
            boolean_mask = tf.equal(A_bool, tf.ones_like(A_bool))
            mask_paddings = tf.ones_like(S) * (-(2 ** 32) + 1)
            S = tf.nn.softmax(
                tf.where(boolean_mask, S, mask_paddings),
                #  name="att_weights",
                axis = -1
            )
            #  if self.is_train_stage:
            #  S = tf.nn.dropout(S, 1.0 - 0.5)
            Xc = tf.matmul(S, X) # B*L*L x B*L*F -> B*L*F
            #  Xc = tf.contrib.layers.bias_add(Xc)
            Xc += X
            Xc = tf.nn.leaky_relu(Xc)

            # <<<<<< ablation_study wo/CGAT <<<<<<<<<<

            ## >>>>>>> ablation_study GCN >>>>>> 
            #  简单GCN，权重即经过归一化的拉普拉斯矩阵矩阵
            #  Z = tf.layers.dense(X, 40, use_bias=False)
            #  Xc = tf.matmul(A, Z)
            #  Xc = tf.nn.leaky_relu(Xc)
            ## <<<<<<< ablation_study GCN <<<<<<

            ## >>>>>>> ablation_study GCN >>>>>> 
            #  加权GCN，余弦相似度卷积，权重即节点之间的余弦相似度
            #  X_norm = tf.nn.l2_normalize(X,dim=2)
            #  S = tf.matmul(X_norm, tf.transpose(X_norm, (0,2,1))) # B*L*L
            #  S = tf.nn.softmax(S, axis=-1)
            #  S = S * A_bool
            #  Xc = tf.matmul(S, X) # B*L*L x B*L*F -> B*L*F
            #  Xc = tf.nn.l2_normalize(Xc,dim=2)
            ## <<<<<<< ablation_study GCN <<<<<<

            ## >>>>>>> ablation_study GCN >>>>>> 
            ## 加权GCN, 概率簇余弦相似度的卷积
            #  X_clu = tf.matmul(A, X) # B*L*L x B*L*1 = B*L*1
            #  X_norm = tf.nn.l2_normalize(X,dim=2)
            #  X_clu_norm = tf.nn.l2_normalize(X_clu,dim=2)
            #  S = tf.matmul(X_clu_norm, tf.transpose(X_norm, (0,2,1))) # B*L*L
            #  S = tf.nn.softmax(S, axis=-1)
            #  S = S * A_bool
            #  Xc = tf.matmul(S, X) # B*L*L x B*L*F -> B*L*F
            #  Xc = tf.nn.l2_normalize(Xc,dim=2)
            ## <<<<<<< ablation_study GCN <<<<<<

            ## >>>>>>> ablation_study GCN >>>>>> 
            ## maybe Leconv: 局部极值卷积网络
            #  term1 = tf.layers.dense(X, 40, use_bias=False) # B*L*F x F*1 -> B*L*1
            #  term2 = tf.layers.dense(X, 40, use_bias=False) # B*L*F x F*1 -> B*L*1
            #  term2 = term2 * tf.reduce_sum(A, -1, keepdims=True) # B*L*1 x B*L*1 = B*L*1
            #  term3 = tf.layers.dense(X, 40, use_bias=False) # B*L*F x F*1 -> B*L*1
            #  term3 = tf.matmul(A, term3) # B*L*L x B*L*1 = B*L*1
            #  Xc = tf.nn.leaky_relu(term1 + term2 - term3, -1) # B*L*1 -> B*L
            ## <<<<<<< ablation_study GCN <<<<<<

            ## >>>>>>> ablation_study GCN >>>>>> 
            ## maybe, 局部注意力极值卷积网络
            #  X_k = tf.layers.dense(X, 40)
            #  f_1 = tf.layers.dense(X_k, 1)  # B*L*F x F*1 -> B*L*1
            #  f_2 = tf.layers.dense(X_k, 1) # B*L*F x F*1 -> B*L*1
            #  S = A_bool * f_1 + A_bool * tf.transpose(f_2, (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
            #  S = tf.nn.leaky_relu(S)
            #  S = tf.nn.softmax(S, axis=-1)
            #  S = S * A_bool
            #  term1 = X
            #  term2 = X * tf.reduce_sum(S, -1, keepdims=True)
            #  term3 = tf.matmul(S, X) # B*L*L x B*L*1 = B*L*1
            #  X = tf.squeeze(term1 + term2 - term3, -1) # B*L*1 -> B*L
            ## <<<<<<< ablation_study GCN <<<<<<


        with tf.name_scope('AnPool'):
            ## QPool Layer  ('cluster_selection'):

            #  #  与target item做query，缺点是最终挑出来的都是与target item相似的item，不具有区分度。 我们同时考虑簇中心的份和相似度得分，将关键锚节点提取。
            #  #  _attention_fcn中的softmax将正负无穷置为0~1，下面的先mask再topk就没有问题
            cluster_score = f_1 + f_2

            ## >>>>>>> ablation_study Pool topk >>>>>> 
            #  分别只考虑‘簇中心得分’和‘query相关得分’
            #  从特征角度，当与自身的相似度越小（即与其它邻域相似度越高），该节点越成为簇的中心，越代表用户的核心兴趣，越该被留下来
            #  cluster_score = f_1
            #  从特征角度，只考虑query的得分
            #  cluster_score = f_2
            ## <<<<<<< ablation_study Pool topk <<<<<<

            ## >>>>>>> ablation_study Pool topk >>>>>> 
            ## 从图结构角度，根据度来确定topk节点
            # 非零元素数都是一样大的，因为阈值太小，所以是完全图，与节点数一致
            # 但最新的结果，差距也不大
            # 用这个，后面会在收敛的时候导致embedding都是一样的（反映在cluster_score都是一样的，即节点数量分之一），然后greater_equal导致并不会缩小图的规模。
            #  cluster_score = tf.cast(tf.count_nonzero(A, -1), tf.float32)
            #  cluster_score = tf.nn.softmax(cluster_score, axis=-1)
            ## <<<<<<< ablation_study Pool topk <<<<<<

            ## >>>>>>> ablation_study Pool topk >>>>>> 
            #  从特征+图结构角度，用LEConv (metrix version)根据自身结构获得topk重要节点，不考虑外部query
            #  term1 = tf.layers.dense(Xc, 1, use_bias=False) # B*L*F x F*1 -> B*L*1
            #  term2 = tf.layers.dense(Xc, 1, use_bias=False) # B*L*F x F*1 -> B*L*1
            #  term2 = term2 * tf.reduce_sum(A, -1, keepdims=True) # B*L*1 x B*L*1 = B*L*1
            #  term3 = tf.layers.dense(Xc, 1, use_bias=False) # B*L*F x F*1 -> B*L*1
            #  term3 = tf.matmul(A, term3) # B*L*L x B*L*1 = B*L*1
            #  #  cluster fitness
            #  cluster_score = tf.squeeze(term1 + term2 - term3, -1) # B*L*1 -> B*L
            #  # sigmoid就是对所有元素执行一个0-1的压缩，并非进行归一化，所以先squeeze还是后squeeze都是一样的。
            #  cluster_score = tf.sigmoid(cluster_score)
            #  # sigmoid/softmax将正负无穷置为0~1，下面的先mask再topk就没有问题
            #  #  cluster_score = tf.nn.softmax(cluster_score, axis=-1) # B*L
            ## <<<<<<< ablation_study Pool topk <<<<<<

            #  对传播得到的锚节点/簇进行weighted readout（只readout筛选后的item，效果更好）
            cluster_fea = tf.reduce_sum(X*tf.expand_dims(cluster_score,-1), 1)
            #  cluster_fea = tf.reduce_sum(Xc*tf.expand_dims(cluster_score,-1), 1) # 和上面差不多
            #  cluster_fea = tf.reduce_sum(Xc*tf.expand_dims(self.real_mask, -1), 1) # 效果极差

            # Multiply X and y to make layer differentiable，这句是tokpool的实现中说的
            # 看看效果如何，好的话后面就直接对X进行sum pooling再输出到最后就行
            #  #  X = X * K.sigmoid(cluster_score)
            #  X2 = X2 * tf.expand_dims(cluster_score, -1)
            #  # 或者直接用cluster_output
            # 效果不好，会影响后面的agru
            #  X2 = cluster_output

            # >>>>>> sagpool 利用gcn对节点进行打分
            #  Z = tf.layers.dense(X, 1, use_bias=False) # B*L*F x F*1 -> B*L*1
            #  # score有可能是负值
            #  node_score = tf.squeeze(tf.matmul(A, Z), -1) # B*L*L x B*L*1 = B*L*1 -> B*L

            # >>>>>> diffpool，训练两个矩阵，一个是特征变换矩阵，一个是分配矩阵
            #  Z = tf.layers.dense(X, 40, use_bias=False)
            #  Z = tf.matmul(A, Z)
            #  # k是每层固定的池化率（对于我们来说不合适，我们需要按每个序列的长度来池化）
            #  S = tf.layers.dense(X, k, use_bias=False)
            #  S = tf.matmul(A, S)
            #  # 对分配矩阵加权
            #  _, alphas = self._attention(self.target_item_embedding, X, var_id = str(k), return_alpha=True)
            #  S = S * tf.expand_dims(alphas, -1)
            #  S = tf.keras.activations.softmax(S, axis=-1)

            #  # TODO 辅助loss：get auxiliary loss
            #  # Link prediction loss
            #  #  第一个正则表示的是辅助链路预测目标。
            #  S_gram = tf.matmul(S, tf.transpose(S, (0,2,1)))
            #  LP_loss = A - S_gram
            #  LP_loss = tf.norm(LP_loss, axis=(-1, -2))
            #  LP_loss = K.mean(LP_loss)
            #  self.auxiliary_loss += LP_loss
            #  # Entropy loss
            #  #  第二个正则表示尽量让每一行的节点分配接近one-hot向量，即一个节点分配到一个cluster内。
            #  # 我们基于attention计算的loss也可以考虑这个
            #  entr = tf.negative(tf.reduce_sum(tf.multiply(S, K.log(S + K.epsilon())), axis=-1))
            #  entr_loss = K.mean(entr, axis=-1)
            #  entr_loss = K.mean(entr_loss)
            #  self.auxiliary_loss += entr_loss

            #  # pooled
            #  X = tf.matmul(tf.transpose(S, (0,2,1)), Z)
            #  A = tf.matmul(
            #      tf.matmul(
            #          tf.transpose(S, (0,2,1)),
            #          A),
            #      S)
            # <<<<<< diffpool，训练两个矩阵，一个是特征变换矩阵，一个是分配矩阵

            # 不用softmax，用min-max归一化，softmax不适合mask场景
            # TODO 上面改写为bool后，这里就不需要了,需要
            #  # 忽略了一个问题，余弦相似度是-1~1的，后面还有mask的过程，所以这里应该先min-max归一化（非余弦相似度计算得到的score也能用），min-max标准化比softmax更适合mask场景
            #  cluster_score = (cluster_score - tf.reduce_min(cluster_score, -1, keepdims=True)) / (tf.reduce_max(cluster_score, -1, keepdims=True) - tf.reduce_min(cluster_score, -1, keepdims=True))
#

            ## 提取topk/topthread节点数
            # 提取ratio比率的高分节点（即，将其余节点mask掉）
            #  num_nodes = tf.reduce_sum(self.mask, 1) # B
            #  to_keep = tf.math.ceil(self.pool_ratio * tf.cast(num_nodes, tf.float32)) # B

            # future 根据最终目标的兴趣序列长度，决定每次选择的topk
            num_nodes = tf.reduce_sum(self.mask, 1) # B
            boolean_pool = tf.greater(num_nodes, self.pool_length)
            to_keep = tf.where(boolean_pool, 
                               tf.cast(self.pool_length + (self.real_sequence_length - self.pool_length)/self.pool_layers*(self.pool_layers-layer-1), tf.int32), 
                               num_nodes)  # B

            ## update mask
            ## 更新topk节点数量的mask，每个graph的topk不同
            cluster_score = cluster_score * self.real_mask # B*L
            if 'kwai' in socket.gethostname():
                sorted_score = tf.contrib.framework.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
            else:
                sorted_score = tf.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
            if 'kwai' in socket.gethostname():
            #  if 'rl3' in socket.gethostname():
                n_graphs = tf.shape(Xc)[0] 
                target_index = tf.stack([tf.range(n_graphs), tf.cast(to_keep, tf.int32)], 1) # B*2
                target_score = tf.gather_nd(sorted_score, target_index) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            else:
                # 为了避免搜到mask的零元素，但缺点在于如果score学到的都是一样的分数，则一个节点也拿不出来了
                #  target_score = tf.gather_nd(sorted_score, tf.expand_dims(tf.cast(to_keep, tf.int32), -1), batch_dims=1)
                target_score = tf.gather_nd(sorted_score, tf.expand_dims(tf.cast(to_keep, tf.int32), -1), batch_dims=1) + K.epsilon()
            #  topk_mask = tf.greater_equal(cluster_score, tf.expand_dims(target_score, -1)) # B*L + B*1 -> B*L
            topk_mask = tf.greater(cluster_score, tf.expand_dims(target_score, -1)) # B*L + B*1 -> B*L
            #  topk_mask = tf.greater(cluster_score, tf.expand_dims(target_score, -1)) # B*L + B*1 -> B*L
            self.mask = tf.cast(topk_mask, tf.int32)
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.sequence_length = tf.reduce_sum(self.mask, 1)

            #  对传播得到的锚节点/簇进行weighted readout（只readout筛选后/mask的node/cluster，效果更好）
            #  cluster_fea = tf.reduce_sum(X*tf.expand_dims(cluster_score,-1)*tf.expand_dims(self.real_mask, -1), 1)

            #  pp1 = tf.reduce_max(tf.reduce_sum(self.mask, 1))
            #  pp2 = tf.shape(self.mask)
            #  pp3 = tf.shape(X)
            #  self.pp1=tf.Print(pp1,["pp1:", pp1]) # 要对应名称，因为执行顺序可能不一样
            #  self.pp2=tf.Print(pp2,["pp2:", pp2])
            #  self.pp3=tf.Print(pp3,["pp3:", pp3])

            #  pp1 = num_nodes
            #  pp2 = to_keep
            #  pp3 = tf.reduce_mean(num_nodes)
            #  pp4 = tf.reduce_mean(to_keep)
            #  pp5 = tf.reduce_sum(self.mask, 1) # B
            #  pp6 = tf.reduce_mean(tf.reduce_sum(self.mask, 1))
            
            #  [pp1:][20 15 16...]
            #  [pp2:][8 6 7...]
            #  [pp3:][24]
            #  [pp4:][10.0196]
            #  [pp5:][20 15 16...]
            #  [pp6:][24]

            #  pp1 = tf.shape(target_index)
            #  pp2 = tf.shape(target_score)
            #  [pp1:][2500 2]
            #  [pp2:][2500]

            #  pp1 = target_score
            #  pp2 = tf.reduce_mean(target_score)
            #  [pp1:][0.0714285746 0.0833333358 0.1...]
            #  [pp2:][0.057143867]

            #  pp1 = target_score
            #  pp2 = tf.reduce_max(cluster_score, axis=-1)
            #  pp3 = tf.reduce_min(cluster_score, axis=-1)
            #  pp4 = tf.reduce_sum(cluster_score, axis=-1)
            #  pp5 = tf.reduce_mean(cluster_score, axis=-1)
            #  [pp1:][0.0833333358 0.0769230798 0.0833333358...]
            #  [pp2:][0.0833333358 0.0769230798 0.0833333358...]
            #  [pp3:][0 0 0...]
            #  [pp4:][1 1.00000012 1...]
            #  [pp5:][0.02] # 1/50

            #  pp1 = tf.sort(cluster_score[0], direction='DESCENDING', axis=-1)
            #  pp1 = tf.sort(A[0][0], direction='DESCENDING', axis=-1)
            #  pp2 = tf.sort(tf.cast(tf.count_nonzero(A, -1), tf.float32)[0], direction='DESCENDING', axis=-1)
            #  pp2 = tf.reduce_sum(cluster_score[0])/tf.cast(num_nodes[0],tf.float32)
            #  pp2 = tf.sort(cluster_score[0], direction='ASCENDING', axis=-1)
            #  pp2 = tf.sort(A[0][0], direction='ASCENDING', axis=-1)
            #  pp3 = pp1[3:6]
            #  pp3 = self.sequence_length[0] # 调用的是其它执行图的数据，所以和num_nodes结果不一致。
            #  pp4 = pp1[7:10]
            #  pp5 = target_score
            #  pp5 = tf.reduce_sum(self.mask, 1)
            #  pp6 = num_nodes[0]

            #  self.pp1=tf.Print(pp1,["pp1:", pp1]) # 要对应名称，因为执行顺序可能不一样
            #  self.pp2=tf.Print(pp2,["pp2:", pp2])
            #  self.pp3=tf.Print(pp3,["pp3:", pp3])
            #  self.pp4=tf.Print(pp4,["pp4:", pp4])
            #  self.pp5=tf.Print(pp5,["pp5:", pp5])
            #  self.pp6=tf.Print(pp6,["pp6:", pp6])
#
            if self.watch_A_S_A:
                #  当一层时，这个地方的所有节点的非零元素还正常
                pp4 = tf.sort(tf.cast(tf.count_nonzero(A, -1), tf.float32)[0], direction='DESCENDING', axis=-1)
                #  pp1 = tf.count_nonzero(A[0])
                pp1 = tf.count_nonzero(tf.count_nonzero(A[0], -1), -1)
                self.pp4=tf.Print(pp4,["pp4:", pp4])
                self.pp1=tf.Print(pp1,["pp1:", pp1]) # 要对应名称，因为执行顺序可能不一样

            #### 将粗化改为度量学习 >>>>>>>>>> 
            ## graph_coarsening
            ## 根据新mask提取分配矩阵，再根据分配矩阵粗化图，确保连通性，而非生硬的mask原矩阵（之前少了一次转置的mask，因为后面还有S的转置）
            #  S = S * tf.expand_dims(self.real_mask, -1)  # B*L*L x B*L*1(mask: B*C*1) = B*L*L(mask: B*C*L)
            # 这里的mask和之前根据A_bool的mask都是必要的
            S = S * tf.expand_dims(self.real_mask, -1) * tf.expand_dims(self.real_mask, -2)

            if self.watch_A_S_A:
                # S的度比A的度分布小，因为在S乘了一个A_bool之后是一致的，但又更新了mask，所以整体上应该更小一些。
                pp5 = tf.sort(tf.cast(tf.count_nonzero(S, -1), tf.float32)[0], direction='DESCENDING', axis=-1)
                self.pp5=tf.Print(pp5,["pp5:", pp5])
                #  pp2 = tf.count_nonzero(S[0])
                pp2 = tf.count_nonzero(tf.count_nonzero(S[0], -1), -1)
                self.pp2=tf.Print(pp2,["pp2:", pp2])

            # 下面执行之后[pp1:][18 17 17...] 变为 [pp5:][21 21 21...](50,50)
            #  A = tf.matmul(tf.matmul(S, A),
            # 这里应该改为A_bool（添加self-loop后的），簇的分配主要看分配矩阵的权重，而非原始图的边权重，只要看边有没有就行了
            A = tf.matmul(tf.matmul(S, A_bool),
                          tf.transpose(S, (0,2,1))) # B*C*L x B*L*L x B*L*C = B*C*C
            # 下面绝对是错的，执行之后变为(50,50)
            #  A = tf.matmul(tf.matmul(tf.transpose(S, (0,2,1)), A),
                          #  S) # B*C*L x B*L*L x B*L*C = B*C*C
            #  A = A * tf.expand_dims(self.real_mask, -1) * tf.expand_dims(self.real_mask, -2)

            # 为了避免越发dense，进行边的提取，利用前边的超参（加了之后，A的节点度就不再完全一样了，即不再是完全图）
            # TODO 不提取看看效果，速度超慢，完全图，太dense了
            A_flatten = tf.reshape(A, [tf.shape(A)[0],-1])
            if 'kwai' in socket.gethostname():
                sorted_A_flatten = tf.contrib.framework.sort(A_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
            else:
                sorted_A_flatten = tf.sort(A_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
            # 上面必须更新sequence_length的长度，不然找到的threshold_score就是0，体现为提取到所有50个节点
            #  num_edges = tf.cast(self.sequence_length * self.sequence_length, tf.float32)
            # 这种边的数量的前提是完全图，所以不合理，很容易取到0元素，所以还是用count_nonzero，对每个batch进行count
            num_edges = tf.cast(tf.count_nonzero(A, [1,2]), tf.float32) # B
            to_keep_edge = tf.cast(tf.math.ceil(num_edges * (1-self.threshold)), tf.int32)
            if 'kwai' in socket.gethostname():
            #  if 'rl3' in socket.gethostname():
                n_graphs = tf.shape(X)[0]
                threshold_index = tf.stack([tf.range(n_graphs), tf.cast(to_keep_edge, tf.int32)], 1) # B*2
                #  threshold_score = tf.batch_gather(sorted_A_flatten, threshold_index) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
                threshold_score = tf.gather_nd(sorted_A_flatten, threshold_index) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            else:
                threshold_score = tf.gather_nd(sorted_A_flatten, tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1), batch_dims=1) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            A = tf.cast(tf.greater_equal(A, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)

            if self.watch_A_S_A:
                # 这里S的节点度的分布应该平均比粗化前的A的节点度的分布要小，刚开始确实是这样，但是后面的时候就基本和A一致了，说明S学成了全1矩阵，也不是，可能是attention学成了一样，然后greater_equal全部取了出来
                #  当一层时，这里所有节点的非零元素就变为了所有节点数，即完全图，所以问题的确出在图粗化的矩阵变换
                # 这个应该比上面的A和S的分布还要小。早起step的时候还正常，后面就每个节点的度都趋向于一样，即完全图了
                pp6 = tf.sort(tf.cast(tf.count_nonzero(A, -1), tf.float32)[0], direction='DESCENDING', axis=-1)
                self.pp6=tf.Print(pp6,["pp6:", pp6])
                #  pp3 = tf.count_nonzero(A[0])
                # 统计非零行/列，1)和上面每个节点的度都是一样的，说明就是完全图；2)和上面的S的非零行是一样的，说明实现无误，问题出在我们的图很小，所以这种处理方式最终相当于三跳相连，如果在thread步上生成的图很dense的话，这里三跳之后就可能是完全图了。
                # 但再设计一个超参不合理，所以这里可以和前面threshold联动，如果前面小（因为取top threshold，所以相当于门槛高），则图稀疏，这里则池化后门槛可以设置的低一些，即top threshold应该取大一点
                # 这样的话，其实对这个超参不敏感了，前后联动，会抵消影响
                pp3 = tf.count_nonzero(tf.count_nonzero(A[0], -1), -1)
                self.pp3=tf.Print(pp3,["pp3:", pp3])
#
            ## >>>>>>> ablation_study wo / weighted gcn >>>>>>
            # 上面的A为0/1，S为0~1；这里S为0/1，A为0~1；
            #  S = tf.ones_like(A)
            #  S = S * tf.expand_dims(self.real_mask, -1)  # B*L*L x B*L*1(mask: B*C*1) = B*L*L(mask: B*C*L)
            #  A = tf.matmul(tf.matmul(S, A),
                          #  tf.transpose(S, (0,2,1))) # B*C*L x B*L*L x B*L*C = B*C*C
            ## <<<<<<< ablation_study weighted gcn <<<<<<

            ## >>>>>>> wo/S, 生硬的mask原矩阵 >>>>>>
            #  A = A * tf.expand_dims(self.real_mask, -1)  # B*L*L x B*L*1(mask: B*C*1) = B*L*L(mask: B*C*L)

            ## <<<<<<<  wo/S, 生硬的mask原矩阵 <<<<<<

        #  return X, A
        return Xc, A, cluster_fea, cluster_score


    def _SURGE(self):

        hparams = self.hparams
        hist_input = tf.concat(
            [self.item_history_embedding, self.cate_history_embedding], 2
        )
        X = hist_input
        self.mask = self.iterator.mask
        self.real_mask = tf.cast(self.mask, tf.float32)
        self.real_sequence_length = tf.reduce_sum(self.mask, 1)
        self.target_item_embedding = self.target_item_embedding[:,:40]
        self.max_n_nodes = int(X.get_shape()[1])

        with tf.name_scope('learnable_graph'):
            ## Hyperparameter study - metric learning的实现

            # 1.sequence (add self loop)
#              A = tf.matmul(X, tf.transpose(X, (0,2,1))) # B*L*L
            #  #  #  A = tf.linalg.band_part(tf.ones_like(A),0,1) # B*L*L
#              A = tf.linalg.band_part(tf.ones_like(A),0,3) # B*L*L

            # 1.complete graph
            #  A = tf.matmul(X, tf.transpose(X, (0,2,1))) # B*L*L
            #  A = tf.ones_like(A) # B*L*L

            # 2. inner-product similarity
            #  A = tf.matmul(X, tf.transpose(X, (0,2,1))) # B*L*L

            # 2.cosine similarity
            # method1
            #  X_norm = tf.sqrt(tf.reduce_sum(tf.square(X), axis=-1)) # B*L*F -> B*L
            #  X_norm = tf.matmul(tf.expand_dims(X_norm, 2),tf.expand_dims(X_norm, 1)) # B*L*1 x B*1*L = B*L*L
            #  A = A/X_norm
            # method2
            #  X_norm = tf.nn.l2_normalize(self.X,dim=2)
            # 应该生成个新变量，因为归一化之后，后面的embedding会影响；或者后面embedding都归一化，试试效果吧
            X_norm = tf.nn.l2_normalize(X,dim=2)
            A = tf.matmul(X_norm, tf.transpose(X_norm, (0,2,1))) # B*L*L
            # 原来没加，会影响后面mask后的排序的图构建（之前cosine效果不稳定的原因就是这个）
            # 不对吧，cosine的阈值就是-1～1
            #  A = tf.nn.softmax(A, axis=-1)
            A = (A - tf.reduce_min(A, -1, keepdims=True)) / (tf.reduce_max(A, -1, keepdims=True) - tf.reduce_min(A, -1, keepdims=True))

            # 再加上时间关系
            #  A = A + tf.linalg.band_part(tf.ones_like(A),0,3) # B*L*L

            # 考虑query的cosine
#              X_fts = tf.concat([X, tf.tile(tf.expand_dims(self.target_item_embedding, 1), [1, self.max_n_nodes, 1])], -1)
            #  #  X_fts = tf.layers.dense(X_tfs, 40, use_bias=False)
            #  X_fts = tf.nn.l2_normalize(X_fts,dim=2)
            #  A = tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1))) # B*L*L
#              A = (A - tf.reduce_min(A, -1, keepdims=True)) / (tf.reduce_max(A, -1, keepdims=True) - tf.reduce_min(A, -1, keepdims=True))

            #  3.weighted cosine similarity
#              self.weighted_tensor = tf.layers.dense(tf.ones([1, 1]), 40, use_bias=False)
            #  X_fts = X * tf.expand_dims(self.weighted_tensor, 0)
            #  #  cosine
            #  X_fts = tf.nn.l2_normalize(X_fts,dim=2)
            #  A = tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1))) # B*L*L
            #  #  # 原来没加，会影响后面mask后的排序的图构建（之前cosine效果不稳定的原因就是这个）
            #  #  A = tf.nn.softmax(A, axis=-1)
            #  # 将softmax换为min-max再看看还有没有梯度消失的问题
#              A = (A - tf.reduce_min(A, -1, keepdims=True)) / (tf.reduce_max(A, -1, keepdims=True) - tf.reduce_min(A, -1, keepdims=True))

            #  3.weighted cosine similarity(fcn)
            #  X_fts = self._fcn_weighted_net(X, 40, 'cosine_weighs')
            #  X_fts = tf.nn.l2_normalize(X_fts,dim=2)
            #  A = tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1))) # B*L*L
            #  A = (A - tf.reduce_min(A, -1, keepdims=True)) / (tf.reduce_max(A, -1, keepdims=True) - tf.reduce_min(A, -1, keepdims=True))


            # 考虑query的weighted cosine
#              X_tar = tf.concat([X, tf.tile(tf.expand_dims(self.target_item_embedding, 1), [1, self.max_n_nodes, 1])], -1)
#              X_fts = tf.layers.dense(X_tar, 40, use_bias=False)
#              # weights
#              self.weighted_tensor = tf.layers.dense(tf.ones([1, 1]), 40, use_bias=False)
#              X_fts = X_fts * tf.expand_dims(self.weighted_tensor, 0)
#              #  cosine
#              X_fts = tf.nn.l2_normalize(X_fts,dim=2)
#              A = tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1))) # B*L*L
#              #  # 原来没加，会影响后面mask后的排序的图构建（之前cosine效果不稳定的原因就是这个）
#  #              A = tf.nn.softmax(A, axis=-1)
#              # 将softmax换为min-max再看看还有没有梯度消失的问题
            #  A = (A - tf.reduce_min(A, -1, keepdims=True)) / (tf.reduce_max(A, -1, keepdims=True) - tf.reduce_min(A, -1, keepdims=True))


            # 4.gat_att
#              X_fts = tf.layers.dense(X, 40, use_bias=False)
            #  f_1 = tf.layers.dense(X_fts, 1)
            #  f_2 = tf.layers.dense(X_fts, 1)
            #  A = f_1 + tf.transpose(f_2, (0,2,1)) # B*L*1 + B*L*1 -> B*L*L
            #  A = tf.nn.leaky_relu(A)
            #  # softmax具有可加性，优于sigmoid
#              A = tf.nn.softmax(A, axis=-1)

            # 4.learnable graph - gat_att_multi
            #  A = []
            #  for i in range(4):
                #  X_fts = tf.layers.dense(X, 40, use_bias=False)
                #  f_1 = tf.layers.dense(X_fts, 1)
                #  f_2 = tf.layers.dense(X_fts, 1)
                #  A += [tf.nn.softmax(tf.nn.leaky_relu(f_1 + tf.transpose(f_2, (0,2,1))), axis=-1)]
            #  A = tf.reduce_mean(tf.stack(A, 0), 0)

            # 5.naive_att
            #  # 可能是向量内积导致难以学习（上面weighted cosine的元素积就没有问题，但是效果不稳定），A中一个节点的所有边的分数完全一样，这样导致在后面构建图的时候用的是greater_equal生成完全图；如果用greater则一条边也连不上
            #  #  X_fts = tf.layers.dense(self.X, 40, use_bias=False, activation=tf.nn.relu)
            #  #  将非线性去掉再看看还有没有梯度消失的问题
            #  X_fts = tf.layers.dense(X, 40, use_bias=False)
            #  #  X_fts = tf.nn.l2_normalize(X_fts,dim=2)
            #  A = tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1))) # B*L*L
            #  #  A = tf.nn.softmax(A, axis=-1)
            #  #  将softmax换为min-max再看看还有没有梯度消失的问题
            #  A = (A - tf.reduce_min(A, -1, keepdims=True)) / (tf.reduce_max(A, -1, keepdims=True) - tf.reduce_min(A, -1, keepdims=True))
#
#              # 5.naive_att(fcn version)
            #  #  X_fts = self._fcn_transform_net(X, [40], scope="fcn_alpha")
            #  X_fts = self._fcn_transform_net_rmact(X, [40], scope="fcn_alpha")
            #  #  X_fts = self._fcn_transform_net_rmact_rmb(X, [40], scope="fcn_alpha")
            #  A = tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1))) # B*L*L
#              A = (A - tf.reduce_min(A, -1, keepdims=True)) / (tf.reduce_max(A, -1, keepdims=True) - tf.reduce_min(A, -1, keepdims=True))


            # 5.naive_att_multi
            #  A = []
            #  for i in range(4):
                #  X_fts = tf.layers.dense(X, 40, use_bias=False, activation=tf.nn.relu)
                #  A += [tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1)))] # B*L*L
            #  A = tf.reduce_mean(tf.stack(A, 0), 0)


            # mask graph to 0/1
            A = A * tf.expand_dims(self.real_mask, -1) * tf.expand_dims(self.real_mask, -2)
            # 余弦相似度就是0-1

            # 计算相对阈值
            #  import pdb; pdb.set_trace()
            A_flatten = tf.reshape(A, [tf.shape(A)[0],-1])
            if 'kwai' in socket.gethostname():
                sorted_A_flatten = tf.contrib.framework.sort(A_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
            else:
                sorted_A_flatten = tf.sort(A_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
            # 1.这种处理方式下，边的数量的前提是完全图，所以不合理，很容易取到0元素，因为用的greater_equal，所以容易构建完全图。还是用count_nonzero，对每个batch进行count
            #  num_edges = tf.cast(self.sequence_length * self.sequence_length, tf.float32)
            # 2.这个问题在于A的矩阵是动态的，边的数量也是动态的，构建图
            num_edges = tf.cast(tf.count_nonzero(A, [1,2]), tf.float32) # B
            to_keep_edge = tf.cast(tf.math.ceil(num_edges * self.threshold), tf.int32)
            if 'kwai' in socket.gethostname():
            #  if 'rl3' in socket.gethostname():
                n_graphs = tf.shape(X)[0]
                threshold_index = tf.stack([tf.range(n_graphs), tf.cast(to_keep_edge, tf.int32)], 1) # B*2
                #  threshold_score = tf.batch_gather(sorted_A_flatten, threshold_index) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
                threshold_score = tf.gather_nd(sorted_A_flatten, threshold_index) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            else:
                threshold_score = tf.gather_nd(sorted_A_flatten, tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1), batch_dims=1) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            #  threshold_score = tf.reduce_min(tf.nn.top_k(A_tmp, threshold_index, sorted=False).values)

            #  pp3 = sorted_A_flatten[0] # should be different
            #  pp4 = sorted_A_flatten[0][-1] # should be 0
            #  pp5 = threshold_score[0] # should be smaller than pp3
            #  self.pp3=tf.Print(pp3,["pp3:", pp3])
            #  self.pp4=tf.Print(pp4,["pp4:", pp4])
            #  self.pp5=tf.Print(pp5,["pp5:", pp5])

            #  A = tf.cast(tf.greater_equal(A, self.threshold), tf.float32)
            #  A = tf.cast(tf.greater_equal(A, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)
            #  self.A = tf.cast(tf.greater(A, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)
            A = tf.cast(tf.greater(A, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)
            #  A = tf.cast(tf.greater_equal(A, tf.expand_dims(threshold_score, -1)), tf.float32)
            # 其实不能用greater_equal，这样的话mask也会变为1，但其实后面asap也有一个mask的过程，所以其实无所谓。
            #  A = tf.cast(tf.greater(A, 0.0), tf.float32)

            # 这个地方节点的度没有问题。
            #  pp2 = tf.sort(tf.cast(tf.count_nonzero(A, -1), tf.float32)[0], direction='DESCENDING', axis=-1)
            #  [pp2:][8 6 5...]
            #  [pp2:][17 16 15...]
            #  self.pp2=tf.Print(pp2,["pp2:", pp2])

            ## >>>>>>> ablation_study weighted gcn >>>>>>
            # mask graph to 0/0~1
            #  A = A * tf.expand_dims(self.real_mask, -1) * tf.expand_dims(self.real_mask, -2)
            #  A_mask = tf.cast(tf.greater_equal(A, self.threshold), tf.float32)
            #  A = A * A_mask
            ## <<<<<<< ablation_study weighted gcn <<<<<<
            self.A = A
            self.X = X

        with tf.name_scope('graph_coarsening'):
            ## Hyperparameter study: ratio, layers and sum

            # 1 layer - 对照实验，用原始特征，即知道核心item后，用原始特征也能取得不错的效果。但问题在于score是根据融合后的特征选择的。
            #  tmp_X = X
            #  X, A = self._interest_fusion(X, A, 0, shared=False, reuse=False)
            #  X = tmp_X

            #  for l in range(2):
                #  reuse = False if l==0 else True
                #  X, A = self._interest_fusion(X, A, layer=l, reuse=reuse)

            #  last
            for l in range(self.pool_layers):
                reuse = False if l==0 else True
                X, A, cluster_fea, alphas = self._interest_fusion(X, A, layer=l, reuse=reuse)

            # sum
            #  readout = []
            #  readout = [X]
            #  for l in range(self.pool_layers):
                #  reuse = False if l==0 else True
                #  X, A, cluster_fea, alphas = self._interest_fusion(X, A, layer=l, reuse=reuse)
                #  readout += [X]
            #  X = sum(readout)

            #  不留存每一步的特征再求和，而是留存每一步的readout，共给最后一层
            #  readout = []
            #  for l in range(self.pool_layers):
            #      reuse = False if l==0 else True
            #      X, A, cluster_fea, alphas = self._interest_fusion(X, A, layer=l, reuse=reuse)
            #      readout += [cluster_fea]
            #  #  cluster_fea = sum(readout)
            #  cluster_fea = tf.concat(readout, 1)


        with tf.name_scope('flatten_sequence'):

            if self.watch_length:
                pp1 = tf.reduce_max(tf.reduce_sum(self.mask, 1))
                pp2 = tf.shape(self.mask)
                pp3 = tf.shape(X)
                self.pp1=tf.Print(pp1,["pp1:", pp1]) # 要对应名称，因为执行顺序可能不一样
                self.pp2=tf.Print(pp2,["pp2:", pp2])
                self.pp3=tf.Print(pp3,["pp3:", pp3])

            # flatten graph to sequence according to initial position
            output_shape = self.mask.get_shape()
            '''
            argsort的参数stable，默认为False，设置为True确保相同元素保序，实际测试时True和False都保序。
            因为乱序尚未实现：https://www.tensorflow.org/api_docs/python/tf/argsort?hl=zh-cn
            sort没有stable参数，相同元素谁前谁后无所谓。
            '''
            if 'kwai' in socket.gethostname():
                sorted_mask_index = tf.contrib.framework.argsort(self.mask, direction='DESCENDING', stable=True, axis=-1) # B*L -> B*L
                sorted_mask = tf.contrib.framework.sort(self.mask, direction='DESCENDING', axis=-1) # B*L -> B*L
            else:
                sorted_mask_index = tf.argsort(self.mask, direction='DESCENDING', stable=True, axis=-1) # B*L -> B*L
                sorted_mask = tf.sort(self.mask, direction='DESCENDING', axis=-1) # B*L -> B*L
            sorted_mask.set_shape(output_shape)
            sorted_mask_index.set_shape(output_shape)
            X = tf.batch_gather(X, sorted_mask_index) # B*L*F  < B*L = B*L*F
            self.mask = sorted_mask
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.sequence_length = tf.reduce_sum(self.mask, 1) # B

            # 按batch 切割graph和特征，加速训练
            self.to_max_length = tf.range(tf.reduce_max(self.sequence_length)) # l
            X = tf.gather(X, self.to_max_length, axis=1) # B*L*F -> B*l*F
            self.mask = tf.gather(self.mask, self.to_max_length, axis=1) # B*L -> B*l
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.sequence_length = tf.reduce_sum(self.mask, 1) # B

            if self.watch_length:
                pp4 = tf.reduce_max(tf.reduce_sum(self.mask, 1))
                pp5 = tf.shape(self.mask)
                pp6 = tf.shape(X)
                self.pp4=tf.Print(pp4,["pp4:", pp4])
                self.pp5=tf.Print(pp5,["pp5:", pp5])
                self.pp6=tf.Print(pp6,["pp6:", pp6])
                #  [pp1:][50]
                #  [pp2:][2500 50]
                #  [pp3:][2500 50 40]
                # 有问题

            # prediction - Our(AGRU 省略求attention，利用输出的cluster score)
            #  _, alphas = self._attention_fcn(self.target_item_embedding, X, 'AGRU', False, return_alpha=True)
            _, final_state = dynamic_rnn_dien(
                VecAttGRUCell(hparams.hidden_size),
                inputs=X,
                att_scores = tf.expand_dims(alphas, -1),
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope="gru"
            )
            model_output = tf.concat([self.target_item_embedding, final_state, cluster_fea, self.target_item_embedding*cluster_fea], 1)
            #  model_output = tf.concat([self.target_item_embedding, final_state, cluster_fea], 1)

            # prediction - DIN
            #  attention_output = self._attention_fcn(self.target_item_embedding, X, 'DIN', False)
            #  att_fea = tf.reduce_sum(attention_output, 1)
            #  self.hist_embedding_sum = tf.reduce_sum(hist_input*tf.expand_dims(self.real_mask, -1), 1)
            #  model_output = tf.concat([self.target_item_embedding, self.hist_embedding_sum, att_fea, cluster_fea], -1)

            # prediction - GRU
            #  rnn_outputs, final_state = dynamic_rnn(
                #  GRUCell(self.hidden_size),
                #  inputs=X,
                #  sequence_length=self.sequence_length,
                #  dtype=tf.float32,
                #  scope="gru",
            #  )
            #  model_output = tf.concat([final_state, cluster_fea, self.target_item_embedding], -1)

            # prediction - AGRU 效果不如我们共享的策略，但是比直接应用agru要好（dien的删减版，不如dien）
            #  _, alphas = self._attention_fcn(self.target_item_embedding, X, 'AGRU', False, return_alpha=True)
            #  _, final_state = dynamic_rnn_dien(
                #  VecAttGRUCell(hparams.hidden_size),
                #  inputs=X,
                #  att_scores = tf.expand_dims(alphas, -1),
                #  sequence_length=self.sequence_length,
                #  dtype=tf.float32,
                #  scope="gru2"
            #  )
            #  self.hist_embedding_sum = tf.reduce_sum(hist_input*tf.expand_dims(self.real_mask, -1), 1)
            #  model_output = tf.concat([self.target_item_embedding, final_state, self.hist_embedding_sum, self.target_item_embedding*self.hist_embedding_sum, cluster_fea], 1)

            # prediction - DIEN 效果不好，我们的兴趣提取层和rnn功能重复，会干扰学习
            #  X = tf.batch_gather(hist_input, sorted_mask_index) # B*L*F  < B*L = B*L*F
            #  rnn_outputs, _ = dynamic_rnn_dien(
            #      GRUCell(hparams.hidden_size),
            #      inputs=X,
            #      sequence_length=self.sequence_length,
            #      dtype=tf.float32,
            #      scope="gru1"
            #  )
            #  _, alphas = self._attention_fcn(self.target_item_embedding, rnn_outputs, 'DIEN', False, return_alpha=True)
            #  _, final_state = dynamic_rnn_dien(
            #      VecAttGRUCell(hparams.hidden_size),
            #      inputs=rnn_outputs,
            #      att_scores = tf.expand_dims(alphas, -1),
            #      sequence_length=self.sequence_length,
            #      dtype=tf.float32,
            #      scope="gru2"
            #  )
            #  self.hist_embedding_sum = tf.reduce_sum(hist_input*tf.expand_dims(self.real_mask, -1), 1)
            #  #  model_output = tf.concat([self.target_item_embedding, final_state, self.hist_embedding_sum, self.target_item_embedding*self.hist_embedding_sum], 1)
            #  model_output = tf.concat([self.target_item_embedding, final_state, self.hist_embedding_sum, self.target_item_embedding*self.hist_embedding_sum, cluster_fea], 1)

        return model_output

#      def _get_loss(self):
        #  """Make loss function, consists of data loss, regularization loss and graph loss
        #
        #  Returns:
        #      obj: Loss value
        #  """
        #
        #  self.data_loss = self._compute_data_loss()
        #  self.regular_loss = self._compute_regular_loss()
        #  #  self.graph_loss = self._compute_graph_loss()
        #  #  self.loss = tf.add(self.data_loss, self.regular_loss)
        #  #  self.loss = self.data_loss + self.regular_loss + self.graph_loss
        #
        #  #  self.LP_loss = self._compute_LP_loss()
        #  #  self.entr_loss = self._compute_entr_loss()
        #
        #  #  self.loss = self.data_loss + self.regular_loss + self.LP_loss + self.entr_loss
        #  #  self.loss = self.data_loss + self.regular_loss + self.entr_loss
        #  self.loss = self.data_loss + self.regular_loss + self.LP_loss
        #
#          #  self.loss = self.data_loss + self.regular_loss + self.graph_loss + self.entr_loss

        return self.loss
#
    def _compute_graph_loss(self):
        """Graph regularization loss"""
        #  discrepancy_loss = tf.reduce_mean(
            #  tf.math.squared_difference(
                #  tf.reshape(self.involved_user_long_embedding, [-1]),
                #  tf.reshape(self.involved_user_short_embedding, [-1])
            #  )
        #  )
        #  discrepancy_loss = -tf.multiply(self.hparams.discrepancy_loss_weight, discrepancy_loss)

        #  graph_loss = 0

        #  L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        #  graph_loss += self.smoothness_ratio * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        #  ones_vec = to_cuda(torch.ones(out_adj.size(-1)), self.device)
        #  graph_loss += -self.degree_ratio * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + Constants.VERY_SMALL_NUMBER)).squeeze() / out_adj.shape[-1]
        #  graph_loss += self.sparsity_ratio * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))

        #  import pdb; pdb.set_trace()
        L = tf.ones_like(self.A) * tf.eye(self.max_n_nodes) * tf.reduce_sum(self.A, -1, keep_dims=True) - self.A

        #  laplacian（效果稍好于上面，但差不多）
        #  D = tf.reduce_sum(self.A, axis=-1) # B*L
        #  D = tf.sqrt(D)[:, None] + K.epsilon() # B*1*L
        #  L = (self.A / D) / tf.transpose(D, perm=(0,2,1)) # B*L*L / B*1*L / B*L*1

        graph_loss = self.smoothness_ratio * tf.trace(tf.matmul(tf.transpose(self.X, (0,2,1)), tf.matmul(L, self.X))) / (self.max_n_nodes*self.max_n_nodes)
        ones_vec = tf.tile(tf.ones([1,self.max_n_nodes]), [tf.shape(self.A)[0], 1]) # B*L
        graph_loss += -self.degree_ratio * tf.squeeze(tf.matmul(tf.expand_dims(ones_vec, 1), tf.log(tf.matmul(self.A, tf.expand_dims(ones_vec,-1)) + K.epsilon())), (-1,-2)) / self.max_n_nodes
        graph_loss += self.sparsity_ratio * tf.reduce_sum(tf.math.pow(self.A, 2), (-1,-2)) / self.max_n_nodes
        graph_loss = tf.reduce_mean(graph_loss)

        return graph_loss

    #  def _attention_gat(self, query, user_embedding, return_alpha=False):
    #  def _attention_gat(self, query, user_embedding, var_id,return_alpha=False):
    def _attention_gat(self, query, user_embedding, var_id, reuse,return_alpha=False):
    #  def _attention_gat(self, S, query, user_embedding, var_id, reuse,return_alpha=False):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item which is regarded as a query in attention operations.
            user_embedding (obj): The output of RNN layers which is regarded as user modeling.

        Returns:
            obj: Weighted sum of user modeling.
        """
        hparams = self.hparams
        #  with tf.variable_scope("attention_fcn"):
        #  with tf.variable_scope("attention_fcn", reuse=reuse):
        #  with tf.variable_scope("attention_fcn"+var_id):
        with tf.variable_scope("attention_fcn"+str(var_id), reuse=reuse):
            query_size = query.shape[-1].value
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            #  scope.reuse_variables()
            attention_mat = tf.get_variable(
                #  name="attention_mat",
                name="attention_mat"+str(var_id),
                shape=[user_embedding.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(user_embedding, attention_mat, [[2], [0]])

            #  att_inputs = tf.layers.dense(user_embedding, 40)
            #  att_inputs = tf.matmul(S, att_inputs)

#              queries = tf.reshape(
                #  #  tf.tile(query, [1, att_inputs.shape[1].value]), tf.shape(att_inputs)
                #  #  https://blog.csdn.net/suiyueruge1314/article/details/105843759
                #  tf.tile(query, [1, tf.shape(att_inputs)[1]]), tf.shape(att_inputs)
#              )
            queries = query
            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
                #  [att_inputs, queries], -1
            )
            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="att_fcn"
            )
            #  att_fnc_output = tf.layers.dense(last_hidden_nn_layer, 1)
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = user_embedding * tf.expand_dims(att_weights, -1)
            if not return_alpha:
                return output
            else:
                return output, att_weights


