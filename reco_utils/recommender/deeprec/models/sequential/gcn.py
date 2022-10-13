# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import socket
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import SequentialBaseModel
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import VecAttGRUCell
from reco_utils.recommender.deeprec.models.sequential.rnn_dien import dynamic_rnn as dynamic_rnn_dien
from tensorflow.keras import backend as K
# remove target
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from tensorflow.nn import dynamic_rnn

## diffpool
from tensorflow.keras import backend as K

__all__ = ["SURGEModel"]


class SURGEModel(SequentialBaseModel):

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization of variables or temp hyperparameters

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
        """
        self.hparams = hparams
        self.relative_threshold = 0.1
        #  self.relative_threshold = 0.3
        self.metric_heads = 1
        self.attention_heads = 1
        self.pool_layers = 1
        self.layer_shared = True
        self.remove_target = False
        self.recent_target = False
        # graph loss
        self.smoothness_ratio = 0.1
        self.degree_ratio = 0.1
        self.sparsity_ratio = 0.1
        # regularization
        #  self.same_mapping_regu, self.single_affiliation_regu, self.relative_position_regu = 1e-7, 1e-7, 1e-7
        self.same_mapping_regu, self.single_affiliation_regu, self.relative_position_regu = 1e-2, 1e-2, 1e-3
        # self.same_mapping_regu, self.single_affiliation_regu, self.relative_position_regu = 1e-3, 1e-3, 1e-5
        #  self.pool_ratio = 0.5
        #  self.pool_ratio = 0.1
        self.pool_ratio = 0.6
        self.pool_length = 10 # taobao
        super().__init__(hparams, iterator_creator, seed=None)


    def _build_seq_graph(self):
        """ SURGE Model: 

            1) Interest graph: Graph construction based on metric learning
            2) Interest fusion and extraction : Graph convolution and graph pooling 
            3) Prediction: Flatten pooled graph to reduced sequence
        """
        X = tf.concat(
            [self.item_history_embedding, self.cate_history_embedding], 2
        )
        X = X + self.position_embedding
        self.mask = self.iterator.mask
        self.float_mask = tf.cast(self.mask, tf.float32)
        self.real_sequence_length = tf.reduce_sum(self.mask, 1)
        # compute recent for matching
        self.recent_k = 1
        #  self.recent_k = 5
        self.real_mask = tf.cast(self.mask, tf.float32)
        self.position = tf.math.cumsum(self.real_mask, axis=1, reverse=True)
        self.recent_mask = tf.logical_and(self.position >= 1, self.position <= self.recent_k)
        self.real_recent_mask = tf.where(self.recent_mask, tf.ones_like(self.recent_mask, dtype=tf.float32), tf.zeros_like(self.recent_mask, dtype=tf.float32))
        self.recent_embedding_mean = tf.reduce_sum(X*tf.expand_dims(self.real_recent_mask, -1), 1)/tf.reduce_sum(self.real_recent_mask, 1, keepdims=True)

        self.max_n_nodes = int(X.get_shape()[1])

        # change to his mean
        #  self.real_mask = tf.cast(self.mask, tf.float32)
        #  self.recent_embedding_mean = tf.reduce_sum(X*tf.expand_dims(self.real_mask, -1), 1)/tf.reduce_sum(self.real_mask, 1, keepdims=True)

        with tf.name_scope('interest_graph'):
            ## Node similarity metric learning 

            # raw graph
#              A = tf.matmul(X, tf.transpose(X, (0,2,1))) # B*L*L
            #  #  A = tf.linalg.band_part(tf.ones_like(A),0,1) # B*L*L
#              A = tf.linalg.band_part(tf.ones_like(A),0,3) # B*L*L

            S = []
            for i in range(self.metric_heads):
                # weighted cosine similarity
                self.weighted_tensor = tf.layers.dense(tf.ones([1, 1]), X.shape.as_list()[-1], use_bias=False)
                # import pdb
                # pdb.set_trace()
                X_fts = X * tf.expand_dims(self.weighted_tensor, 0)
                #  X_fts = X
                X_fts = tf.nn.l2_normalize(X_fts,dim=2)
                S_one = tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1))) # B*L*L

                #  S_one = A + S_one

                #  min-max normalization for mask
                #  S_min = tf.reduce_min(S_one, -1, keepdims=True)
                #  S_max = tf.reduce_max(S_one, -1, keepdims=True)
                #  S_one = (S_one - S_min) / (S_max - S_min)

                # Euclidean distance
#                  p1 = tf.reduce_sum(tf.square(X), -1, keepdims=True)
                #  dist = tf.sqrt(tf.add(p1, tf.transpose(p1, (0,2,1))) - 2 * tf.matmul(X, tf.transpose(X, (0,2,1))))
                #  # 高斯核将距离转换为相似度，之后元素为0-1
#                  S_one = tf.exp(-1*dist)

                # Maha distance
#                  M_w = tf.get_variable(
                #      name="maha_weight",
                #      shape=[X.shape.as_list()[-1], X.shape.as_list()[-1]],
                #      initializer=self.initializer,
                #  )
                #  #  X_fts = tf.tensordot(X, M_w, [[2], [0]])
                #  # 根据转置矩阵乘法展开，要求M_w左乘X
                #  X_fts = tf.tensordot(X, M_w, [[-1], [-1]])
                #  p1 = tf.reduce_sum(tf.square(X_fts), -1, keepdims=True)
                #  dist = tf.sqrt(tf.add(p1, tf.transpose(p1, (0,2,1))) - 2 * tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1))))
                #  # 高斯核将距离转换为相似度，之后元素为0-1
#                  S_one = tf.exp(-1*dist)

                #  S_one = A + S_one

                S += [S_one]
            S = tf.reduce_mean(tf.stack(S, 0), 0)
            # mask invalid nodes
            S = S * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)


            ## Graph sparsification via seted sparseness
            S_flatten = tf.reshape(S, [tf.shape(S)[0],-1])
            if 'kwai' in socket.gethostname():
                sorted_S_flatten = tf.contrib.framework.sort(S_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
            else:
                sorted_S_flatten = tf.sort(S_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
            # relative ranking strategy of the entire graph
            num_edges = tf.cast(tf.count_nonzero(S, [1,2]), tf.float32) # B
            to_keep_edge = tf.cast(tf.math.ceil(num_edges * self.relative_threshold), tf.int32)
            if 'kwai' in socket.gethostname():
                threshold_index = tf.stack([tf.range(tf.shape(X)[0]), tf.cast(to_keep_edge, tf.int32)], 1) # B*2
                threshold_score = tf.gather_nd(sorted_S_flatten, threshold_index) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            else:
                threshold_score = tf.gather_nd(sorted_S_flatten, tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1), batch_dims=1) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            A = tf.cast(tf.greater(S, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)

            #  A = S



        with tf.name_scope('interest_fusion_extraction'):
            for l in range(self.pool_layers):
                reuse = False if l==0 else True
                #  X, A, graph_readout, alphas = self._interest_fusion_extraction(X, A, layer=l, reuse=reuse)
                X, A, graph_readout, alphas = self._interest_fusion_extraction_new(X, A, layer=l, reuse=reuse)
                #  X = self.normalize(X)


        with tf.name_scope('prediction'):
#              # flatten pooled graph to reduced sequence
#              output_shape = self.mask.get_shape()
#              if 'kwai' in socket.gethostname():
#                  sorted_mask_index = tf.contrib.framework.argsort(self.mask, direction='DESCENDING', stable=True, axis=-1) # B*L -> B*L
#                  sorted_mask = tf.contrib.framework.sort(self.mask, direction='DESCENDING', axis=-1) # B*L -> B*L
#              else:
#                  sorted_mask_index = tf.argsort(self.mask, direction='DESCENDING', stable=True, axis=-1) # B*L -> B*L
#                  sorted_mask = tf.sort(self.mask, direction='DESCENDING', axis=-1) # B*L -> B*L
#              sorted_mask.set_shape(output_shape)
#              sorted_mask_index.set_shape(output_shape)
#              X = tf.batch_gather(X, sorted_mask_index) # B*L*F  < B*L = B*L*F
#              # change alphas
#              #  alphas = tf.batch_gather(alphas, sorted_mask_index) # B*L  < B*L = B*L
#              self.mask = sorted_mask
#              self.reduced_sequence_length = tf.reduce_sum(self.mask, 1) # B
#  #
#              # cut useless sequence tail per batch
#              self.to_max_length = tf.range(tf.reduce_max(self.reduced_sequence_length)) # l
#              X = tf.gather(X, self.to_max_length, axis=1) # B*L*F -> B*l*F
#              # change alphas
#              alphas = tf.gather(alphas, self.to_max_length, axis=1) # B*L -> B*l
#              self.mask = tf.gather(self.mask, self.to_max_length, axis=1) # B*L -> B*l
#              self.float_mask = tf.cast(self.mask, tf.float32)
#              self.reduced_sequence_length = tf.reduce_sum(self.mask, 1) # B
#
            # rm sort
            self.reduced_sequence_length = tf.reduce_sum(self.mask, 1) # B

            # fix length, k = 10
            #  self.reduced_sequence_length = tf.fill([tf.shape(self.mask)[0]],10)

            # use cluster score as attention weights in AUGRU 
            if not self.remove_target: 
                #  if not self.recent_target:
                    #  _, alphas = self._attention_fcn(self.target_item_embedding, X, 'AGRU', False, return_alpha=True)
                #  else:
                    #  _, alphas = self._attention_fcn(self.recent_embedding_mean, X, 'AGRU', False, return_alpha=True)
                _, final_state = dynamic_rnn_dien(
                    VecAttGRUCell(self.hparams.hidden_size),
                    inputs=X,
                    att_scores = tf.expand_dims(alphas, -1),
                    sequence_length=self.reduced_sequence_length,
                    dtype=tf.float32,
                    scope="gru"
                )

                #  _, final_state = dynamic_rnn(
                    #  GRUCell(self.hparams.hidden_size),
                    #  inputs=X,
                    #  sequence_length=self.reduced_sequence_length,
                    #  dtype=tf.float32,
                    #  scope="gru",
                #  )
#

                #  self.mask = tf.ones([tf.shape(self.mask)[0], 10])
                #  attention_output, alphas = self._attention_fcn(self.target_item_embedding, X, 'Att-last', False, return_alpha=True)
                #  final_state = tf.reduce_sum(attention_output, 1)

                #  final_state = tf.reduce_mean(X, 1)

            else:
                _, final_state = dynamic_rnn(
                    GRUCell(self.hparams.hidden_size),
                    inputs=X,
                    sequence_length=self.reduced_sequence_length,
                    dtype=tf.float32,
                    scope="gru",
                )

            #  graph_readout = tf.reduce_sum(X*tf.expand_dims(alphas,-1)*tf.expand_dims(self.float_mask, -1), 1)

            #  #  self.recent_k = 5
            #  self.recent_k = 1
            #  self.real_mask = tf.cast(self.mask, tf.float32)
            #  self.position = tf.math.cumsum(self.real_mask, axis=1, reverse=True)
            #  self.recent_mask = tf.logical_and(self.position >= 1, self.position <= self.recent_k)
            #  self.real_recent_mask = tf.where(self.recent_mask, tf.ones_like(self.recent_mask, dtype=tf.float32), tf.zeros_like(self.recent_mask, dtype=tf.float32))
            #  self.recent_embedding_mean = tf.reduce_sum(X*tf.expand_dims(self.real_recent_mask, -1), 1)/tf.reduce_sum(self.real_recent_mask, 1, keepdims=True)

            if not self.remove_target: 
                if not self.recent_target:
                    model_output = tf.concat([final_state, graph_readout, self.target_item_embedding, graph_readout*self.target_item_embedding], 1)
                    #  model_output = tf.concat([final_state, graph_readout, self.target_item_embedding], 1)
                else:
                    #  model_output = tf.concat([final_state, graph_readout, self.target_item_embedding, graph_readout*self.recent_embedding_mean], 1)
                    #  model_output = tf.concat([final_state, graph_readout, self.target_item_embedding, self.recent_embedding_mean ,graph_readout*self.recent_embedding_mean], 1)
                    model_output = tf.concat([final_state, self.target_item_embedding, self.recent_embedding_mean], 1)
                    #  model_output = tf.concat([final_state, self.target_item_embedding], 1)
            else:
                #  model_output = tf.concat([final_state, graph_readout, self.target_item_embedding, graph_readout], 1)
                #  model_output = tf.concat([final_state, graph_readout, self.target_item_embedding], 1)
                model_output = tf.concat([final_state, self.target_item_embedding], 1)


        return model_output

  
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


    def _interest_fusion_extraction(self, X, A, layer, reuse):
        """Interest fusion and extraction via graph convolution and graph pooling 

        Args:
            X (obj): Node embedding of graph
            A (obj): Adjacency matrix of graph
            layer (obj): Interest fusion and extraction layer
            reuse (obj): Reusing variable W in query operation 

        Returns:
            X (obj): Aggerated cluster embedding 
            A (obj): Pooled adjacency matrix 
            graph_readout (obj): Readout embedding after graph pooling
            cluster_score (obj): Cluster score for AUGRU in prediction layer

        """
        with tf.name_scope('interest_fusion'):
            ## cluster embedding
            A_bool = tf.cast(tf.greater(A, 0), A.dtype)
            A_bool = A_bool * (tf.ones([A.shape.as_list()[1],A.shape.as_list()[1]]) - tf.eye(A.shape.as_list()[1])) + tf.eye(A.shape.as_list()[1])
            D = tf.reduce_sum(A_bool, axis=-1) # B*L
            D = tf.sqrt(D)[:, None] + K.epsilon() # B*1*L
            A = (A_bool / D) / tf.transpose(D, perm=(0,2,1)) # B*L*L / B*1*L / B*L*1
            Xq = tf.matmul(A, tf.matmul(A, X)) # B*L*F

            Xc = []
            for i in range(self.attention_heads):
                ## cluster- and query-aware attention
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(Xq, X, 'f1_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                    if not self.remove_target:
                        if not self.recent_target:
                            _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                        else:
                            _, f_2 = self._attention_fcn(self.recent_embedding_mean, X, 'f2_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                if self.layer_shared:
                    _, f_1 = self._attention_fcn(Xq, X, 'f1_shared'+'_'+str(i), reuse, return_alpha=True)
                    if not self.remove_target:
                        if not self.recent_target:
                            _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_shared'+'_'+str(i), reuse, return_alpha=True)
                        else:
                            _, f_2 = self._attention_fcn(self.recent_embedding_mean, X, 'f2_shared'+'_'+str(i), reuse, return_alpha=True)

                ## graph attentive convolution
                if not self.remove_target:
                    E = A_bool * tf.expand_dims(f_1,1) + A_bool * tf.transpose(tf.expand_dims(f_2,1), (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
                else:
                    E = A_bool * tf.expand_dims(f_1,1) 
                E = tf.nn.leaky_relu(E)
                boolean_mask = tf.equal(A_bool, tf.ones_like(A_bool))
                mask_paddings = tf.ones_like(E) * (-(2 ** 32) + 1)
                E = tf.nn.softmax(
                    tf.where(boolean_mask, E, mask_paddings),
                    axis = -1
                )
                Xc_one = tf.matmul(E, X) # B*L*L x B*L*F -> B*L*F
                Xc_one = tf.layers.dense(Xc_one, 40, use_bias=False)
                #  Xc_one = self.normalize(Xc_one)
                Xc_one += X
                Xc += [tf.nn.leaky_relu(Xc_one)]
            Xc = tf.reduce_mean(tf.stack(Xc, 0), 0)

            #  Xc = self.normalize(Xc)

        with tf.name_scope('interest_extraction'):
            ## cluster fitness score 
            Xq = tf.matmul(A, tf.matmul(A, Xc)) # B*L*F
            cluster_score = []
            for i in range(self.attention_heads):
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(Xq, Xc, 'f1_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
                    if not self.remove_target:
                        if not self.recent_target:
                            _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
                        else:
                            _, f_2 = self._attention_fcn(self.recent_embedding_mean, Xc, 'f2_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
                if self.layer_shared:
                    _, f_1 = self._attention_fcn(Xq, Xc, 'f1_shared'+'_'+str(i), True, return_alpha=True)
                    if not self.remove_target:
                        if not self.recent_target:
                            _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_shared'+'_'+str(i), True, return_alpha=True)
                        else:
                            _, f_2 = self._attention_fcn(self.recent_embedding_mean, Xc, 'f2_shared'+'_'+str(i), True, return_alpha=True)
                if not self.remove_target:
                    cluster_score += [f_1 + f_2]
                else:
                    cluster_score += [f_1]
            cluster_score = tf.reduce_mean(tf.stack(cluster_score, 0), 0)
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))
            mask_paddings = tf.ones_like(cluster_score) * (-(2 ** 32) + 1)
            cluster_score = tf.nn.softmax(
                tf.where(boolean_mask, cluster_score, mask_paddings),
                axis = -1
            )

            ## graph pooling
            num_nodes = tf.reduce_sum(self.mask, 1) # B
            boolean_pool = tf.greater(num_nodes, self.pool_length)
            to_keep = tf.where(boolean_pool, 
                               tf.cast(self.pool_length + (self.real_sequence_length - self.pool_length)/self.pool_layers*(self.pool_layers-layer-1), tf.int32), 
                               num_nodes)  # B
            cluster_score = cluster_score * self.float_mask # B*L
            if 'kwai' in socket.gethostname():
                sorted_score = tf.contrib.framework.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
                target_index = tf.stack([tf.range(tf.shape(Xc)[0]), tf.cast(to_keep, tf.int32)], 1) # B*2
                target_score = tf.gather_nd(sorted_score, target_index) + K.epsilon() # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            else:
                sorted_score = tf.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
                target_score = tf.gather_nd(sorted_score, tf.expand_dims(tf.cast(to_keep, tf.int32), -1), batch_dims=1) + K.epsilon() # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            topk_mask = tf.greater(cluster_score, tf.expand_dims(target_score, -1)) # B*L + B*1 -> B*L
            self.mask = tf.cast(topk_mask, tf.int32)
            self.float_mask = tf.cast(self.mask, tf.float32)
            self.reduced_sequence_length = tf.reduce_sum(self.mask, 1)

            ## ensure graph connectivity 
            E = E * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)
            A = tf.matmul(tf.matmul(E, A_bool),
                          tf.transpose(E, (0,2,1))) # B*C*L x B*L*L x B*L*C = B*C*C
            ## graph readout 
            graph_readout = tf.reduce_sum(Xc*tf.expand_dims(cluster_score,-1)*tf.expand_dims(self.float_mask, -1), 1)

        return Xc, A, graph_readout, cluster_score

    def _interest_fusion_extraction_new(self, X, A, layer, reuse):
        with tf.name_scope('interest_fusion'):
            ## cluster embedding
            A_bool = tf.cast(tf.greater(A, 0), A.dtype)
            A_bool = A_bool * (tf.ones([A.shape.as_list()[1],A.shape.as_list()[1]]) - tf.eye(A.shape.as_list()[1])) + tf.eye(A.shape.as_list()[1])
            D = tf.reduce_sum(A_bool, axis=-1) # B*L
            D = tf.sqrt(D)[:, None] + K.epsilon() # B*1*L
            A = (A_bool / D) / tf.transpose(D, perm=(0,2,1)) # B*L*L / B*1*L / B*L*1
            Xq = tf.matmul(A, tf.matmul(A, X)) # B*L*F

            #  self.A = A
            #  self.X = X

            # gcn
            #  D = tf.reduce_sum(A, axis=1)  # degree for each atom
            #  d += np.spacing(np.array(0, W.dtype))
            #  d = 1 / np.sqrt(d)
            #  D = np.diag(d.squeeze())  # D^{-1/2}
            #  I = np.identity(d.size, dtype=W.dtype)
            #  L = I - D * W * D

            # multi-head gat
            Xc = []
            node_score = []
            for i in range(self.attention_heads):
                ## cluster- and query-aware attention
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(Xq, X, 'f1_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                    if not self.remove_target:
                        if not self.recent_target:
                            _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                        else:
                            _, f_2 = self._attention_fcn(self.recent_embedding_mean, X, 'f2_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                if self.layer_shared:
                    _, f_1 = self._attention_fcn(Xq, X, 'f1_shared'+'_'+str(i), reuse, return_alpha=True)
                    if not self.remove_target:
                        if not self.recent_target:
                            _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_shared'+'_'+str(i), reuse, return_alpha=True)
                        else:
                            _, f_2 = self._attention_fcn(self.recent_embedding_mean, X, 'f2_shared'+'_'+str(i), reuse, return_alpha=True)

                #  node_score += [f_1 + f_2]
                #  node_score += [0.5*f_1 + 0.5*f_2]
                #  node_score += [f_2]

                ## graph attentive convolution
                if not self.remove_target:
                    E = A_bool * tf.expand_dims(f_1,1) + A_bool * tf.transpose(tf.expand_dims(f_2,1), (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
                    #  E = A_bool * tf.expand_dims(0.1 * f_1,1) + A_bool * tf.transpose(tf.expand_dims(0.9 * f_2,1), (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
                    #  E = A_bool * tf.expand_dims(0.3 * f_1,1) + A_bool * tf.transpose(tf.expand_dims(0.7 * f_2,1), (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
                    #  E = A_bool * tf.expand_dims(0.5 * f_1,1) + A_bool * tf.transpose(tf.expand_dims(0.5 * f_2,1), (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
                else:
                    E = A_bool * tf.expand_dims(f_1,1)
                E = tf.nn.leaky_relu(E)
                boolean_mask = tf.equal(A_bool, tf.ones_like(A_bool))
                mask_paddings = tf.ones_like(E) * (-(2 ** 32) + 1)
                E = tf.nn.softmax(
                    tf.where(boolean_mask, E, mask_paddings),
                    axis = -1
                )
                Xc_one = tf.matmul(E, X) # B*L*L x B*L*F -> B*L*F
                Xc_one = tf.layers.dense(Xc_one, self.hparams.hidden_size, use_bias=False)
                #  Xc_one = self.normalize(Xc_one)
                Xc_one += X
                Xc += [tf.nn.leaky_relu(Xc_one)]
                #  Xc += [Xc_one]
            Xc = tf.reduce_mean(tf.stack(Xc, 0), 0)
            #  node_score = tf.reduce_mean(tf.stack(node_score, 0), 0)
#
            #  Xc = X

            #  _, f_11 = self._attention_fcn(Xq, X, 'f11_shared'+'_'+str(i), reuse, return_alpha=True)
            #  _, f_22 = self._attention_fcn(self.target_item_embedding, X, 'f22_shared'+'_'+str(i), reuse, return_alpha=True)
            #  _, f_11 = self._attention_fcn(Xq, X, 'f11_shared', reuse, return_alpha=True)
            #  _, f_22 = self._attention_fcn(self.target_item_embedding, X, 'f22_shared', reuse, return_alpha=True)
            #  node_score = f_11 + f_22
            #  node_score = f_22

            # perform softmax on node score
            #  boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            #  boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))
            #  mask_paddings = tf.ones_like(node_score) * (-(2 ** 32) + 1)
            #  node_score = tf.nn.softmax(
                #  tf.where(boolean_mask, node_score, mask_paddings),
                #  name="att_weights",
            #  )

            #  Xc = self.normalize(Xc)

            # new gat
            #  Xc, E = self.gat(X, 1, A_bool)

            # None
            #  Xc = X
            #  _, node_score = self._attention_fcn(self.target_item_embedding, Xc, 'AGRU', False, return_alpha=True)
            #  _, node_score = self._attention_fcn(self.target_item_embedding, X, 'AGRU', False, return_alpha=True)
            #  node_score = f_1 + f_2
            #  node_score = 0.3*f_1 + 0.7*f_2
            #  node_score = 0.5*f_1 + 0.5*f_2
            node_score = f_2
            graph_readout = tf.reduce_sum(Xc*tf.expand_dims(node_score,-1)*tf.expand_dims(tf.cast(self.mask, tf.float32), -1), 1)
            #  graph_readout = tf.reduce_sum(X*tf.expand_dims(node_score,-1)*tf.expand_dims(tf.cast(self.mask, tf.float32), -1), 1)

            # new node score
            #  node_score = f_2
            #  _, node_score = self._attention_fcn(self.target_item_embedding, X, 'AGRU', False, return_alpha=True)
            #  _, node_score = self._attention_fcn(self.target_item_embedding, Xc, 'AGRU', False, return_alpha=True)

#          with tf.name_scope('interest_extraction'):
#              ## cluster fitness score
#              Xq = tf.matmul(A, tf.matmul(A, Xc)) # B*L*F
#
#              cluster_score = []
#              for i in range(self.attention_heads):
#                  if not self.layer_shared:
#                      _, f_1 = self._attention_fcn(Xq, Xc, 'f1_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
#                      #  _, f_1 = self._attention_fcn(Xq, Xc, 'f1_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
#                      if not self.remove_target:
#                          if not self.recent_target:
#                              _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
#                              #  _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
#                          else:
#                              _, f_2 = self._attention_fcn(self.recent_embedding_mean, Xc, 'f2_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
#                              #  _, f_2 = self._attention_fcn(self.recent_embedding_mean, Xc, 'f2_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
#                  if self.layer_shared:
#                      _, f_1 = self._attention_fcn(Xq, Xc, 'f1_shared'+'_'+str(i), True, return_alpha=True)
#                      #  _, f_1 = self._attention_fcn(Xq, Xc, 'f1_shared'+'_'+str(i), False, return_alpha=True)
#                      if not self.remove_target:
#                          if not self.recent_target:
#                              _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_shared'+'_'+str(i), True, return_alpha=True)
#                              #  _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_shared'+'_'+str(i), False, return_alpha=True)
#                          else:
#                              _, f_2 = self._attention_fcn(self.recent_embedding_mean, Xc, 'f2_shared'+'_'+str(i), True, return_alpha=True)
#                              #  _, f_2 = self._attention_fcn(self.recent_embedding_mean, Xc, 'f2_shared'+'_'+str(i), False, return_alpha=True)
#                  if not self.remove_target:
#                      cluster_score += [f_1 + f_2]
#                  else:
#                      cluster_score += [f_1]
#              cluster_score = tf.reduce_mean(tf.stack(cluster_score, 0), 0)
#
#  #              # cosine distance
#              #  # f1
#              #  #求模
#              #  Xc_norm = tf.sqrt(tf.reduce_sum(tf.square(Xc), axis=2))
#              #  Xq_norm = tf.sqrt(tf.reduce_sum(tf.square(Xq), axis=2))
#              #  #内积
#              #  Xc_Xq = tf.reduce_sum(tf.multiply(Xc, Xq), axis=2)
#              #  f1_cos = Xc_Xq / (Xc_norm * Xq_norm)
#              #  # f2
#              #  #求模
#              #  Xt = tf.expand_dims(self.target_item_embedding, 1)
#              #  Xt_norm = tf.sqrt(tf.reduce_sum(tf.square(Xt), axis=2))
#              #  #内积
#              #  Xc_Xt = tf.reduce_sum(tf.multiply(Xc, Xt), axis=2)
#              #  f2_cos = Xc_Xt / (Xc_norm * Xt_norm)
#              #
#              #  cluster_score = f1_cos + f2_cos
#  #              #  cluster_score = f2_cos
#
#              boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))
#              mask_paddings = tf.ones_like(cluster_score) * (-(2 ** 32) + 1)
#              cluster_score = tf.nn.softmax(
#                  tf.where(boolean_mask, cluster_score, mask_paddings),
#                  axis = -1
#              )
#
#              ## graph pooling
#              num_nodes = tf.reduce_sum(self.mask, 1) # B
#              boolean_pool = tf.greater(num_nodes, self.pool_length)
#              to_keep = tf.where(boolean_pool,
#                                 tf.cast(self.pool_length + (self.real_sequence_length - self.pool_length)/self.pool_layers*(self.pool_layers-layer-1), tf.int32),
#                                 num_nodes)  # B
#              cluster_score = cluster_score * self.float_mask # B*L
#              if 'kwai' in socket.gethostname():
#                  sorted_score = tf.contrib.framework.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
#                  target_index = tf.stack([tf.range(tf.shape(Xc)[0]), tf.cast(to_keep, tf.int32)], 1) # B*2
#                  target_score = tf.gather_nd(sorted_score, target_index) + K.epsilon() # indices[:-1]=(B) + data[indices[-1]=() --> (B)
#              else:
#                  sorted_score = tf.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
#                  target_score = tf.gather_nd(sorted_score, tf.expand_dims(tf.cast(to_keep, tf.int32), -1), batch_dims=1) + K.epsilon() # indices[:-1]=(B) + data[indices[-1]=() --> (B)
#              topk_mask = tf.greater(cluster_score, tf.expand_dims(target_score, -1)) # B*L + B*1 -> B*L
#              self.mask = tf.cast(topk_mask, tf.int32)
#              self.float_mask = tf.cast(self.mask, tf.float32)
#              self.reduced_sequence_length = tf.reduce_sum(self.mask, 1)
#
#              ## ensure graph connectivity
#              E = E * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)
#              A = tf.matmul(tf.matmul(E, A_bool),
#                            tf.transpose(E, (0,2,1))) # B*C*L x B*L*L x B*L*C = B*C*C
#              ## graph readout
#              graph_readout = tf.reduce_sum(Xc*tf.expand_dims(cluster_score,-1)*tf.expand_dims(self.float_mask, -1), 1)

            #  graph_readout = tf.reduce_sum(Xc*tf.expand_dims(node_score,-1)*tf.expand_dims(self.float_mask, -1), 1)

            #  Xc,A, cluster_score = self._diffpool(Xc, A)
            #  Xc,A, cluster_score = self._diffpool(X, Xc, A)
            #  Xc,A, cluster_score = self._diffpool(X, Xc, A, node_score)
            Xc, A, cluster_score = self._diffpool(Xc, A, node_score)
            #  Xc, A, cluster_score = self._diffpool(self.normalize(Xc), A, node_score)
            #  Xc, A, cluster_score = self._diffpool(X, A, node_score)
            #  Xc,A = self._diffpool(self.normalize(Xc), A)
            #  cluster_score = node_score

            # None
            #  _, node_score = self._attention_fcn(self.target_item_embedding, Xc, 'AGRU', False, return_alpha=True)
            #  graph_readout = tf.reduce_sum(Xc*tf.expand_dims(node_score,-1)*tf.expand_dims(tf.cast(self.mask, tf.float32), -1), 1)
            #  graph_readout = tf.reduce_sum(Xc*tf.expand_dims(cluster_score,-1)*tf.expand_dims(tf.cast(self.mask, tf.float32), -1), 1)
            #  graph_readout = tf.reduce_sum(Xc*tf.expand_dims(tf.cast(self.mask, tf.float32), -1), 1)

        return Xc, A, graph_readout, cluster_score
        #  return Xc, A, graph_readout, node_score


    def gat(self, X, num_heads, A_bool):

        #  X = self.normalize(X)

        Q = tf.layers.conv1d(X, 40, 1) # B*L*F
        K = tf.layers.conv1d(X, 40, 1)
        V = tf.layers.conv1d(X, 40, 1, use_bias=False)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0) # B*LxH*F/H
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)

        att_score_ = tf.reduce_sum(Q_ * K_, axis=-1) # B*LxH

        A_bool_ = tf.tile(A_bool, [1, num_heads, 1]) # B*L*L -> B*LxH*L
        E = A_bool_ * tf.expand_dims(att_score_,-1)  # B*LxH*L x B*LxH*1 -> B*LxH*L
        E = tf.nn.leaky_relu(E)
        boolean_mask_ = tf.equal(A_bool_, tf.ones_like(A_bool_))
        mask_paddings = tf.ones_like(E) * (-(2 ** 32) + 1) # B*LxH*L
        E = tf.nn.softmax(
            tf.where(boolean_mask_, E, mask_paddings), # B*LxH*L
            axis = -1
        )
        #  Xc_one = tf.matmul(E, X) # B*L*L x B*L*F -> B*L*F
        h_ = tf.matmul(E, V_) # B*LxH*L x B*L*F/H -> B*LxH*F/H
        #  Xc_one = tf.layers.dense(Xc_one, 40, use_bias=False)

        h = tf.concat(tf.split(h_, num_heads, axis=0), axis=-1) # B*L*F

        h += X
        h = tf.nn.leaky_relu(h)

        h = self.normalize(h)

        return h, E


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

    def _get_loss(self):
        """Make loss function, consists of data loss, regularization loss and graph loss

        Returns:
            obj: Loss value
        """

        self.data_loss = self._compute_data_loss()
        self.regular_loss = self._compute_regular_loss()

        ## graph loss
        #  self.graph_loss = self._compute_graph_loss()
        #  self.loss = self.data_loss + self.regular_loss + self.graph_loss

        ## diffpool loss

        self.loss = self.data_loss + self.auxiliary_loss + self.regular_loss

        #  self.LP_loss = self._compute_LP_loss()
        #  self.entr_loss = self._compute_entr_loss()

        #  self.loss = self.data_loss + self.regular_loss + self.LP_loss + self.entr_loss
        #  self.loss = self.data_loss + self.regular_loss + self.entr_loss
        #  self.loss = self.data_loss + self.regular_loss + self.LP_loss

        #  self.loss = self.data_loss + self.regular_loss + self.graph_loss + self.auxiliary_loss

        return self.loss


    #  def _diffpool(self, X, A):
    #  def _diffpool(self, X, Xc, A):
    #  def _diffpool(self, X, Xc, A, node_score):
    def _diffpool(self, X, A, node_score):

        hparams = self.hparams
        with tf.name_scope('diffpool'):
#              # generate initial graph and normalize it
            #  # product
            #  A = tf.matmul(X, tf.transpose(X, (0,2,1)))
            #
            #  # attention（但之前已经用过attention了，所以这里不需要再用了吧）
            #  #  f_1 = tf.layers.conv1d(X, 1, 1)
            #  #  f_2 = tf.layers.conv1d(X, 1, 1)
            #  #  A = f_1 * tf.transpose(f_2, perm=(0,2,1))
            #  #  A = A + tf.eye(hparams.max_seq_length, dtype=A.dtype)
            #
            #  # raw
            #  #  A = tf.linalg.band_part(tf.ones([X.shape[0], hparams.max_seq_length, hparams.max_seq_length], tf.float32),0,1)
            #  A = tf.linalg.band_part(tf.ones_like(A),0,1)
            #
            #  D = tf.reduce_sum(A, axis=-1)
            #  D = tf.sqrt(D)[:, None] + K.epsilon()
            #  A = (A / D) / tf.transpose(D, perm=(0,2,1))

            # excute diffpool
            #  for k in [30]:
            #  for k in [20]:
            #  for k in [10]:
            #  for k in [5]:
            #  for k in [30, 10]:
            #  for k in [20, 5, 1]:
            #  for k in [30, 15, 5]:
            #  for k in [40, 25, 10]:
            k = 30
            for _ in range(self.pool_layers):
                # Update node embeddings
                Z = tf.layers.dense(X, 40, use_bias=False)
                #  Z = tf.matmul(A, Z)
                #  Z = tf.nn.leaky_relu(Z)

                #  Z = X

                # Compute cluster assignment matrix
                S = tf.layers.dense(X, k, use_bias=False) # B*L*F -> B*L*k
                S = tf.matmul(A, S) # B*L*L x B*L*k -> B*L*k
                #  _, alphas = self._attention_fcn(self.target_item_embedding, X, 'pool_att_'+str(k), False, return_alpha=True)
                #  S = S * tf.expand_dims(alphas, -1)

                ## not considering mask
                #  S = tf.keras.activations.softmax(S, axis=-1)

                ## considering mask

                # 1. via length
#                  num_nodes = tf.reduce_sum(self.mask, 1) # B
                #  boolean_pool = tf.greater(num_nodes, self.pool_length)
                #  to_keep = tf.where(boolean_pool,
                #                     tf.cast(self.pool_length + (self.real_sequence_length - self.pool_length)/self.pool_layers*(self.pool_layers-layer-1), tf.int32),
#                                     num_nodes)  # B

                # 2.via ratio
                num_nodes = tf.cast(tf.reduce_sum(self.mask, 1), tf.float32) # B
                node_position = tf.cast(tf.math.cumsum(self.mask, axis=1), tf.float32) # B*L 
                boolean_pool = tf.less(node_position, tf.expand_dims(self.pool_ratio*num_nodes, -1)) # B*L + B*1 -> B*L(k)
                # if k not 50
                boolean_pool = tf.batch_gather(boolean_pool, tf.tile(tf.expand_dims(tf.range(k), 0), [tf.shape(self.mask)[0], 1]))

                mask_paddings = tf.ones_like(S) * (-(2 ** 32) + 1) # B*L*L(k)
                S = tf.nn.softmax(
                    tf.where(tf.tile(tf.expand_dims(boolean_pool, 1), [1, tf.shape(self.mask)[1], 1]), S, mask_paddings),
                    axis = -1
                ) # B*1*L(k) + B*L*L(k) + B*L*(k) -> B*L*L(k)

                self.mask = tf.cast(boolean_pool, tf.int32) # B*L(k)

                #  position flatten (Differentiable?)
#                  cluster_position = tf.reduce_sum(tf.transpose(S, (0,2,1)) * tf.expand_dims(node_position, 1), -1) # B*L(k)*L + B*1*L -> B*L(k)*L -> B*L(k)
                #  if 'kwai' in socket.gethostname():
                #      sorted_cluster_index = tf.contrib.framework.argsort(node_position, direction='ASCENDING', stable=True, axis=-1) # B*L(k) -> B*L(k)
                #      sorted_cluster = tf.contrib.framework.sort(node_position, direction='ASCENDING', axis=-1) # B*L(k) -> B*L(k)
                #  else:
                #      sorted_cluster_index = tf.argsort(node_position, direction='ASCENDING', stable=True, axis=-1) # B*L(k) -> B*L(k)
                #      sorted_cluster = tf.sort(node_position, direction='ASCENDING', axis=-1) # B*L(k) -> B*L(k)
#                  S = tf.transpose(tf.batch_gather(tf.transpose(S, (0,2,1)), sorted_cluster_index), (0,2,1)) # B*L(k)*L  < B*L(k) -> B*L(k)*L -> B*L*L(k)

                ## Auxiliary pooling loss
                self.auxiliary_loss = 0.0
                # Link prediction loss
                S_gram = tf.matmul(S, tf.transpose(S, (0,2,1)))
                LP_loss = A - S_gram #  LP_loss = A/tf.norm(A) - S_gram/tf.norm(S_gram)
                LP_loss = tf.norm(LP_loss, axis=(-1, -2))
                LP_loss = K.mean(LP_loss)
                self.auxiliary_loss += self.same_mapping_regu*LP_loss
                # Entropy loss
                entr = tf.negative(tf.reduce_sum(tf.multiply(S, K.log(S + K.epsilon())), axis=-1))
                entr_loss = K.mean(entr, axis=-1)
                entr_loss = K.mean(entr_loss)
                self.auxiliary_loss += self.single_affiliation_regu*entr_loss
                # Position loss
                Pn = tf.math.cumsum(tf.ones([tf.shape(S)[0], 1, tf.shape(S)[1]]), axis=-1) # node position encoding:
                Pc = tf.math.cumsum(tf.ones([tf.shape(S)[0], 1, tf.shape(S)[2]]), axis=-1) # cluster position encoding:
                position_loss = tf.matmul(Pn, S) - Pc
                position_loss = tf.norm(position_loss, axis=(-1, -2))
                position_loss = K.mean(position_loss)
                self.auxiliary_loss += self.relative_position_regu*position_loss

                # pooled
                X = tf.matmul(tf.transpose(S, (0,2,1)), Z) # B*L(k)*L x B*L*F -> B*L(k)*F
                cluster_score = tf.squeeze(tf.matmul(tf.transpose(S, (0,2,1)), tf.expand_dims(node_score, -1)), -1) # B*L(k)*L x B*L*1 -> B*L(k)*1 -> B*L(k)
                ## perform softmax on cluster score
                #  boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))
                #  mask_paddings = tf.ones_like(cluster_score) * (-(2 ** 32) + 1) # B*L*L(k)
                #  cluster_score = tf.nn.softmax(
                    #  tf.where(boolean_mask, cluster_score, mask_paddings),
                    #  axis = -1
                #  )

                X = X*tf.expand_dims(cluster_score, -1)
                A = tf.matmul(
                    tf.matmul(
                        tf.transpose(S, (0,2,1)), 
                        A), 
                    S) # B*L(k)*L x B*L*L x B*L*L(k) -> B*L(k)*L(k)

            # when last k is 1 (only one node)
            #  att_fea = tf.squeeze(X, 1)

        return X, A, cluster_score

