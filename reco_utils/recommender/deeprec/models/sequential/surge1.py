# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from tensorflow.keras import backend as K
import socket
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import (
    VecAttGRUCell,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.models.sequential.rnn_dien import dynamic_rnn as dynamic_rnn_dien

__all__ = ["GCNModel"]


class GCNModel(SequentialBaseModel):

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization of variables or temp hyperparameters

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
        """
        self.threshold = 0.3 
        self.pool_ratio = 0.4 
        self.pool_layers = 2
        self.query_shared = True
        super().__init__(hparams, iterator_creator, seed=None)


    def _build_seq_graph(self):
        """ SURGE Model: 

            1) learnable graph: metric learning
            2) graph coarsening: propogation and pooling 
            3) reduced sequence: position flatten
        """
        hparams = self.hparams
        X = tf.concat(
            [self.item_history_embedding, self.cate_history_embedding], 2
        )
        self.mask = self.iterator.mask
        self.real_mask = tf.cast(self.mask, tf.float32)
        self.sequence_length = tf.reduce_sum(self.mask, 1)
        self.max_n_nodes = int(X.get_shape()[1])

        with tf.name_scope('learnable_graph'):
            # weighted cosine similarity
            weighted_tensor = tf.layers.dense(tf.ones([1, 1]), hparams.hidden_size, use_bias=False)
            X_fts = X * tf.expand_dims(weighted_tensor, 0)
            X_fts = tf.nn.l2_normalize(X_fts,dim=2)
            A = tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1))) # B*L*L
            A = tf.nn.softmax(A, axis=-1)
            A = A * tf.expand_dims(self.real_mask, -1) * tf.expand_dims(self.real_mask, -2)

            # conctruct graph according to the threshold
            A_flatten = tf.reshape(A, [tf.shape(A)[0],-1])
            if 'kwai' in socket.gethostname():
                sorted_A_flatten = tf.contrib.framework.sort(A_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
            else:
                sorted_A_flatten = tf.sort(A_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
            num_edges = tf.cast(self.sequence_length * self.sequence_length, tf.float32)
            to_keep_edge = tf.cast(tf.math.ceil(num_edges * self.threshold), tf.int32)
            threshold_score = tf.gather_nd(sorted_A_flatten, tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1), batch_dims=1) 
            A = tf.cast(tf.greater_equal(A, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)

        with tf.name_scope('graph_coarsening'):
            readout = []
            for l in range(self.pool_layers):
                reuse = False if l==0 else True
                X, A = self._graph_coarsening(X, A, layer=l, reuse=reuse)
                readout += [X]

        with tf.name_scope('flatten_sequence'):
            # flatten graph to sequence according to the initial position
            output_shape = self.mask.get_shape()
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

            # split useless sequence tail per batch 
            self.to_max_length = tf.range(tf.reduce_max(self.sequence_length)) # l
            X = tf.gather(X, self.to_max_length, axis=1) # B*L*F -> B*l*F
            self.mask = tf.gather(self.mask, self.to_max_length, axis=1) # B*L -> B*l
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.sequence_length = tf.reduce_sum(self.mask, 1) # B

            # prediction
            attention_output, alphas = self._attention_fcn(self.target_item_embedding, X, 'output', False, return_alpha=True)
            att_fea = tf.reduce_sum(attention_output, 1)
            _, final_state = dynamic_rnn_dien(
                VecAttGRUCell(hparams.hidden_size),
                inputs=X,
                att_scores = tf.expand_dims(alphas, -1),
                sequence_length=self.sequence_length,
                dtype=tf.float32,
                scope="gru"
            )
            tf.summary.histogram('GRU_Final_State', final_state)

        model_output = tf.concat([self.target_item_embedding, final_state, att_fea, self.target_item_embedding*att_fea], 1)
        tf.summary.histogram("model_output", model_output)
        return model_output


    def _graph_coarsening(self, X, A, layer, reuse):
        """Graph Coarsening used to reduce the graph scale

        Args:
            X (obj): node embedding in graph
            A (obj): adjacency matrix of graph
            layer (obj): graph coarsening layer
            reuse (obj): reusing attention W in query operation 

        Returns:
            X (obj): aggerated cluster embedding 
            A (obj): pooled adjacency matrix 
        """

        with tf.name_scope('CGAT'):
            # add self-loop
            A_bool = tf.cast(tf.greater(A, 0), A.dtype)
            A = A_bool * (tf.ones([self.max_n_nodes,self.max_n_nodes]) - tf.eye(self.max_n_nodes)) + tf.eye(self.max_n_nodes)

            # normalize graph
            D = tf.reduce_sum(A, axis=-1) # B*L
            D = tf.sqrt(D)[:, None] + K.epsilon() # B*1*L
            A = (A / D) / tf.transpose(D, perm=(0,2,1)) # B*L*L / B*1*L / B*L*1

            # attention propogation
            X_q = tf.matmul(A, X) # B*L*F
            X_q = tf.layers.dense(X_q, self.hparams.hidden_size)
            X_k = tf.layers.dense(X, self.hparams.hidden_size)
            f_1 = tf.layers.dense(X_q, 1)  # B*L*F x F*1 -> B*L*1
            f_2 = tf.layers.dense(X_k, 1) # B*L*F x F*1 -> B*L*1
            S = A_bool * f_1 + A_bool * tf.transpose(f_2, (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
            S = tf.nn.leaky_relu(S)
            S = tf.nn.softmax(S, axis=-1) * A_bool
            X = tf.matmul(S, X) # B*L*L x B*L*F -> B*L*F

        with tf.name_scope('QPool'):
            if not self.query_shared:
                _, cluster_score = self._attention_fcn(self.target_item_embedding, X, layer, False, return_alpha=True)
            if self.query_shared:
                _, cluster_score = self._attention_fcn(self.target_item_embedding, X, 'shared', reuse, return_alpha=True)
            num_nodes = tf.reduce_sum(self.mask, 1) # B
            n_graphs = tf.shape(X)[0] 
            to_keep = tf.math.ceil(self.pool_ratio * tf.cast(num_nodes, tf.float32)) # B

            ## update mask
            cluster_score = cluster_score * self.real_mask # B*L
            if 'kwai' in socket.gethostname():
                sorted_score = tf.contrib.framework.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
            else:
                sorted_score = tf.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
            target_score = tf.gather_nd(sorted_score, tf.expand_dims(tf.cast(to_keep, tf.int32), -1), batch_dims=1)
            topk_mask = tf.greater_equal(cluster_score, tf.expand_dims(target_score, -1)) # B*L + B*1 -> B*L
            self.mask = tf.cast(topk_mask, tf.int32)
            self.real_mask = tf.cast(self.mask, tf.float32)

            ## ensure graph connectivity 
            S = S * tf.expand_dims(self.real_mask, -1) * tf.expand_dims(self.real_mask, -2)
            A = tf.matmul(tf.matmul(S, A),
                          tf.transpose(S, (0,2,1))) # B*C*L x B*L*L x B*L*C = B*C*C

        return X, A

    def _attention_fcn(self, query, key_embedding, var_id, reuse, return_alpha=False):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item which is regarded as a query in attention operations.
            key_embedding (obj): The embedding of history items which is regarded as keys in attention operations.

        Returns:
            obj: Weights and weighted sum of key embedding.
        """
        hparams = self.hparams
        with tf.variable_scope("attention_fcn"+str(var_id), reuse=reuse):
            query_size = query.shape[1].value
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            attention_mat = tf.get_variable(
                name="attention_mat"+str(var_id),
                shape=[key_embedding.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(key_embedding, attention_mat, [[2], [0]])

            queries = tf.reshape(
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
            output = key_embedding * tf.expand_dims(att_weights, -1)
            if not return_alpha:
                return output
            else:
                return output, att_weights


