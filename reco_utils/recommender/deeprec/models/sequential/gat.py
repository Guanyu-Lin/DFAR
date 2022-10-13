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

__all__ = ["GATModel"]


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
        super().__init__(hparams, iterator_creator, seed=None)


    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        super(GCNModel, self)._build_embedding()

        ## concat id and cate embedding to form node embedding
        item2cate = self.get_item2cate()
        item2cate[0] = 0 # for all unseen item and cate
        for i in range(self.item_vocab_length):
            if i not in item2cate:
                item2cate[i] = 0 # for unseen item in trainset
        sorted_cate = [i[1] for i in sorted(item2cate.items(), reverse=False, key=lambda x:x[0])]
        new_cate_embedding = tf.nn.embedding_lookup(self.cate_lookup, sorted_cate)
        self.item_lookup = tf.concat([self.item_lookup, new_cate_embedding], axis=1)

        self._full_batch_gat()
        print('full-batch gat done')

    def _build_seq_graph(self):
        #  output = self._sum()
        #  output = self._din()
        output = self._gru()
        return output

    def _sum(self):
        with tf.name_scope('sum'):
            #  hist_input = tf.concat(
                #  [self.item_history_embedding, self.cate_history_embedding], 2
            #  )
            # remove cate for gat
            hist_input = self.item_history_embedding
            self.target_item_embedding = self.target_item_embedding[:,:40]
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
            # remove cate for gat
            hist_input = self.item_history_embedding
            self.target_item_embedding = self.target_item_embedding[:,:40]
            self.hist_embedding_sum = tf.reduce_sum(hist_input, 1)
            self.mask = self.iterator.mask

            attention_output = self._attention_fcn(self.target_item_embedding, hist_input)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)

        model_output = tf.concat([self.target_item_embedding, self.hist_embedding_sum, att_fea], -1)
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
            # remove cate for gat
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

            model_output = tf.concat([final_state, self.target_item_embedding], 1)
            tf.summary.histogram("model_output", model_output)
            return model_output
       
    def print_statistics(self, X, string):
        print('>'*10 + string + '>'*10 )
        print('Shape', X.shape)
        print('Average interactions', X.sum(1).mean(0).item())
        nonzero_row_indice, nonzero_col_indice = X.nonzero()
        unique_nonzero_row_indice = np.unique(nonzero_row_indice)
        unique_nonzero_col_indice = np.unique(nonzero_col_indice)
        print('Non-zero rows', float(len(unique_nonzero_row_indice))/float(X.shape[0]))
        print('Non-zero columns', float(len(unique_nonzero_col_indice))/float(X.shape[1]))
        print('Matrix density', float(len(nonzero_row_indice))/float((X.shape[0]*X.shape[1])))
        print('True Average interactions', float(len(nonzero_row_indice))/float(X.shape[0]))


    def get_R(self):

        R_ui = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        R_ii = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)
        train_file = self.train_file
        self.userdict = load_dict(self.hparams.user_vocab)
        print(len(self.userdict))
        self.itemdict = load_dict(self.hparams.item_vocab)
        print(len(self.itemdict))
        with open(train_file) as f_train:
            last_uid = None
            train_lines = f_train.readlines()
            line_num = len(train_lines)
            line_count = 1
            for l in train_lines:
                if len(l) == 0: break
                l = l.strip('\n')
                units = l.strip().split("\t")
                uid = units[1]
                uid = self.userdict[uid]
                #  uid = self.userdict[units[1]] if units[1] in self.userdict else 0
                target_item = units[2]
                item_hist_list = units[5].split(",")

                if uid != last_uid and last_uid is not None:

                    for i in last_item_hist_list:
                        i = self.itemdict[i]
                        R_ui[int(uid), int(i)] = 1.


                    for i,j in zip(last_item_hist_list[:-1], last_item_hist_list[1:]):
                        i = self.itemdict[i]
                        j = self.itemdict[j]
                        R_ii[int(i), int(j)] = 1.
                        R_ii[int(j), int(i)] = 1.

                    k = self.itemdict[last_target_item]
                    R_ii[int(j), int(k)] = 1.
                    R_ii[int(k), int(j)] = 1.

                if line_count == line_num:

                    for i in item_hist_list:
                        i = self.itemdict[i]
                        #  i = self.itemdict[i] if i in self.itemdict else 0
                        R_ui[int(uid), int(i)] = 1.

                    for i,j in zip(item_hist_list[:-1], item_hist_list[1:]):
                        i = self.itemdict[i]
                        j = self.itemdict[j]
                        #  i = self.itemdict[i] if i in self.itemdict else 0
                        #  j = self.itemdict[j] if i in self.itemdict else 0
                        R_ii[int(i), int(j)] = 1.
                        R_ii[int(j), int(i)] = 1.

                    k = self.itemdict[target_item]
                    R_ii[int(j), int(k)] = 1.
                    R_ii[int(k), int(j)] = 1.

                last_uid = uid
                last_item_hist_list = item_hist_list
                last_target_item = target_item
                line_count += 1

        return R_ui, R_ii

    def creat_item2cate(self):
        train_file = self.train_file
        self.userdict = load_dict(self.hparams.user_vocab)
        print(len(self.userdict))
        self.itemdict = load_dict(self.hparams.item_vocab)
        print(len(self.itemdict))
        self.catedict = load_dict(self.hparams.cate_vocab)
        print(len(self.catedict))
        with open(train_file) as f_train:
            last_uid = None
            train_lines = f_train.readlines()
            line_num = len(train_lines)
            line_count = 1
            i2c = {}
            for l in train_lines:
                if len(l) == 0: break
                l = l.strip('\n')
                units = l.strip().split("\t")
                uid = units[1]
                uid = self.userdict[uid]
                #  uid = self.userdict[units[1]] if units[1] in self.userdict else 0
                target_item = units[2]
                target_cate = units[3]
                item_hist_list = units[5].split(",")
                cate_hist_list = units[6].split(",")

                if uid != last_uid and last_uid is not None:

                    t_i = self.itemdict[last_target_item]
                    t_c = self.catedict[last_target_cate]
                    i2c[t_i] = t_c

                    for i,c in zip(last_item_hist_list, last_cate_hist_list):
                        i = self.itemdict[i]
                        c = self.catedict[c]
                        i2c[i] = c

                if line_count == line_num:

                    t_i = self.itemdict[target_item]
                    t_c = self.catedict[target_cate]
                    i2c[t_i] = t_c

                    for i,c in zip(item_hist_list, cate_hist_list):
                        i = self.itemdict[i]
                        c = self.catedict[c]
                        i2c[i] = c

                last_uid = uid
                last_target_item = target_item
                last_target_cate = target_cate
                last_item_hist_list = item_hist_list
                last_cate_hist_list = cate_hist_list
                line_count += 1

        return i2c

    def get_item2cate(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        try:
            t1 = time()
            item2cate = load_dict(self.path + '/item2cate')
            print('already load item2cate', len(item2cate), time() - t1)

        except Exception:
            item2cate = self.creat_item2cate()
            cPickle.dump(item2cate, open(self.path + '/item2cate', "wb"))

        return item2cate
 
    def get_adj_mat(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        try:
            t1 = time()
            adj_mat_short = sp.load_npz(self.path + '/s_adj_mat_short.npz')
            norm_adj_mat_short = sp.load_npz(self.path + '/s_norm_adj_mat_short.npz')
            mean_adj_mat_short = sp.load_npz(self.path + '/s_mean_adj_mat_short.npz')
            print('already load adj matrix short', adj_mat_short.shape, time() - t1)
            adj_mat_ui = sp.load_npz(self.path + '/s_adj_mat_ui.npz')
            norm_adj_mat_ui = sp.load_npz(self.path + '/s_norm_adj_mat_ui.npz')
            mean_adj_mat_ui = sp.load_npz(self.path + '/s_mean_adj_mat_ui.npz')
            print('already load adj matrix ui', adj_mat_ui.shape, time() - t1)
            adj_mat_ii = sp.load_npz(self.path + '/s_adj_mat_ii.npz')
            norm_adj_mat_ii = sp.load_npz(self.path + '/s_norm_adj_mat_ii.npz')
            mean_adj_mat_ii = sp.load_npz(self.path + '/s_mean_adj_mat_ii.npz')
            print('already load adj matrix ii', adj_mat_ii.shape, time() - t1)

        except Exception:
            self.R_ui, self.R_ii = self.get_R()
            self.print_statistics(self.R_ui, 'ui matrix')
            self.print_statistics(self.R_ui.T, 'iu matrix')
            self.print_statistics(self.R_ii, 'ii matrix')

            adj_mat_short, norm_adj_mat_short, mean_adj_mat_short = self.create_adj_mat_short(self.R_ui, self.R_ii)
            adj_mat_ui, norm_adj_mat_ui, mean_adj_mat_ui = self.create_adj_mat_ui(self.R_ui)
            adj_mat_ii, norm_adj_mat_ii, mean_adj_mat_ii = self.create_adj_mat_ii(self.R_ii)
            sp.save_npz(self.path + '/s_adj_mat_short.npz', adj_mat_short)
            sp.save_npz(self.path + '/s_norm_adj_mat_short.npz', norm_adj_mat_short)
            sp.save_npz(self.path + '/s_mean_adj_mat_short.npz', mean_adj_mat_short)

            sp.save_npz(self.path + '/s_adj_mat_ui.npz', adj_mat_ui)
            sp.save_npz(self.path + '/s_norm_adj_mat_ui.npz', norm_adj_mat_ui)
            sp.save_npz(self.path + '/s_mean_adj_mat_ui.npz', mean_adj_mat_ui)

            sp.save_npz(self.path + '/s_adj_mat_ii.npz', adj_mat_ii)
            sp.save_npz(self.path + '/s_norm_adj_mat_ii.npz', norm_adj_mat_ii)
            sp.save_npz(self.path + '/s_mean_adj_mat_ii.npz', mean_adj_mat_ii)

        try:
            pre_adj_mat_short = sp.load_npz(self.path + '/s_pre_adj_mat_short.npz')
            pre_adj_mat_ui = sp.load_npz(self.path + '/s_pre_adj_mat_ui.npz')
            pre_adj_mat_ii = sp.load_npz(self.path + '/s_pre_adj_mat_ui.npz')
        except Exception:
            adj_mat_short=adj_mat_short
            rowsum = np.array(adj_mat_short.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()

            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_short = d_mat_inv.dot(adj_mat_short)
            norm_adj_short = norm_adj_short.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat_short = norm_adj_short.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_short.npz', pre_adj_mat_short)

            adj_mat_ui=adj_mat_ui
            rowsum = np.array(adj_mat_ui.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()

            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_ui = d_mat_inv.dot(adj_mat_ui)
            norm_adj_ui = norm_adj_ui.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat_ui = norm_adj_ui.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_ui.npz', pre_adj_mat_ui)

            adj_mat_ii=adj_mat_ii
            rowsum = np.array(adj_mat_ii.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()

            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_ii = d_mat_inv.dot(adj_mat_ii)
            norm_adj_ii = norm_adj_ii.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat_ii = norm_adj_ii.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_ii.npz', pre_adj_mat_ii)

        return adj_mat_ii, norm_adj_mat_ii, mean_adj_mat_ii, pre_adj_mat_ii

    def get_adj_cooc(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        try:
            t1 = time()
            adj_mat_cooc = sp.load_npz(self.path + '/s_adj_mat_cooc.npz')
            norm_adj_mat_cooc = sp.load_npz(self.path + '/s_norm_adj_mat_cooc.npz')
            mean_adj_mat_cooc = sp.load_npz(self.path + '/s_mean_adj_mat_cooc.npz')
            print('already load adj matrix cooc', adj_mat_cooc.shape, time() - t1)

            adj_mat_cooc2 = sp.load_npz(self.path + '/s_adj_mat_cooc2.npz')
            norm_adj_mat_cooc2 = sp.load_npz(self.path + '/s_norm_adj_mat_cooc2.npz')
            mean_adj_mat_cooc2 = sp.load_npz(self.path + '/s_mean_adj_mat_cooc2.npz')
            print('already load adj matrix cooc in 2order', adj_mat_cooc2.shape, time() - t1)

        except Exception:
            self.R_ui, self.R_ii = self.get_R()
            self.print_statistics(self.R_ui, 'ui matrix')
            self.print_statistics(self.R_ui.T, 'iu matrix')
            self.print_statistics(self.R_ii, 'ii matrix')

            #  adj_mat_cooc, norm_adj_mat_cooc, mean_adj_mat_cooc, adj_mat_cooc2, norm_adj_mat_cooc2, mean_adj_mat_cooc2 = self.create_adj_mat_cooc(self.R_ui)
            adj_mat_cooc, norm_adj_mat_cooc, mean_adj_mat_cooc, adj_mat_cooc2, norm_adj_mat_cooc2, mean_adj_mat_cooc2 = self.create_adj_mat_cooc(self.R_ii)
            sp.save_npz(self.path + '/s_adj_mat_cooc.npz', adj_mat_cooc)
            sp.save_npz(self.path + '/s_norm_adj_mat_cooc.npz', norm_adj_mat_cooc)
            sp.save_npz(self.path + '/s_mean_adj_mat_cooc.npz', mean_adj_mat_cooc)

            sp.save_npz(self.path + '/s_adj_mat_cooc2.npz', adj_mat_cooc2)
            sp.save_npz(self.path + '/s_norm_adj_mat_cooc2.npz', norm_adj_mat_cooc2)
            sp.save_npz(self.path + '/s_mean_adj_mat_cooc2.npz', mean_adj_mat_cooc2)

        try:
            pre_adj_mat_cooc = sp.load_npz(self.path + '/s_pre_adj_mat_cooc.npz')
            pre_adj_mat_cooc22222222 = sp.load_npz(self.path + '/s_pre_adj_mat_cooc2.npz')

        except Exception:
            rowsum = np.array(adj_mat_cooc.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()

            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_cooc = d_mat_inv.dot(adj_mat_cooc)
            norm_adj_cooc = norm_adj_cooc.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat_cooc = norm_adj_cooc.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_cooc.npz', pre_adj_mat_cooc)

            rowsum = np.array(adj_mat_cooc2.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()

            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_cooc2 = d_mat_inv.dot(adj_mat_cooc2)
            norm_adj_cooc2 = norm_adj_cooc2.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat_cooc2 = norm_adj_cooc2.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_cooc2.npz', pre_adj_mat_cooc2)


        return adj_mat_cooc, norm_adj_mat_cooc, mean_adj_mat_cooc, pre_adj_mat_cooc, adj_mat_cooc2, norm_adj_mat_cooc2, mean_adj_mat_cooc2, pre_adj_mat_cooc2

    def create_adj_mat_short(self, R_base, R_side):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R_base = R_base.tolil()
        R_side = R_side.tolil()
        # prevent memory from overflowing
        for i in range(5):
            adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
            R_base[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]
            adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
            R_base[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T
            adj_mat[self.n_users+int(self.n_items*i/5.0):self.n_users+int(self.n_items*(i+1.0)/5), self.n_users:] =\
            R_side[int(self.n_items*i/5.0):int(self.n_items*(i+1.0)/5)]
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        self.print_statistics(adj_mat, 'adj matrix')

        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def create_adj_mat_ui(self, R):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R.tolil()
        # prevent memory from overflowing
        for i in range(5):
            adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]
            adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        
        self.print_statistics(adj_mat, 'adj matrix ui')
            
        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        
        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


    def create_adj_mat_ii(self, R):
        t1 = time()
        adj_mat = R.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        
        self.print_statistics(adj_mat, 'adj matrix ii')
            
        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)
        
        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
  
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

    def normalized_adj_single(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    def _full_batch_gat(self):

        # graph
        self.n_users = self.user_vocab_length
        self.n_items = self.item_vocab_length
        plain_adj, norm_adj, mean_adj, pre_adj = self.get_adj_mat()
        self.print_statistics(norm_adj, 'normed adj matrix ')
        biases = self.preprocess_adj_bias(norm_adj)
        self.print_statistics(biases, 'normed adj matrix biases')
        bias_in = self._convert_sp_mat_to_sp_tensor(biases.tocsr())

        # node feature
        ftr_in = self.item_lookup
        ftr_in = tf.expand_dims(ftr_in, axis=0)

        # hyperparameters
        output_dim = self.hparams.item_embedding_dim + self.hparams.cate_embedding_dim
        nb_nodes = norm_adj.shape[0]
        is_train = None
        attn_drop = 0.0
        ffd_drop = 0.0
        #  hid_units = [5, 5] # numbers of hidden units per each attention head in each layer
        #  n_heads = [8, 8] # additional entry for the output layer
        hid_units = [40, 40] # numbers of hidden units per each attention head in each layer
        n_heads = [1, 1] # additional entry for the output layer
        #  hid_units = [40] # numbers of hidden units per each attention head in each layer
        #  n_heads = [1] # additional entry for the output layer
        residual = False
        nonlinearity = tf.nn.elu

        # gat core
        ftr_out = self.inference(ftr_in, output_dim, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
        ftr_out = tf.squeeze(ftr_out,0)
        self.item_lookup = ftr_out
        return ftr_out

    def preprocess_adj_bias(self, adj):
        num_nodes = adj.shape[0]
        adj = adj + sp.eye(num_nodes)  # self-loop
        adj[adj > 0.0] = 1.0
        if not sp.isspmatrix_coo(adj):
            adj = adj.tocoo()
        adj = adj.astype(np.float32)
        indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
        # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
        #  return indices, adj.data, adj.shape
        return adj

    def sp_attn_head(self, seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
        with tf.name_scope('sp_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)

            seq_fts = tf.layers.dense(seq, out_sz, use_bias=False)

            # simplest self-attention possible
            f_1 = tf.layers.dense(seq_fts, 1)
            f_2 = tf.layers.dense(seq_fts, 1) 
            f_1 = tf.reshape(f_1, (nb_nodes, 1))
            f_2 = tf.reshape(f_2, (nb_nodes, 1))
            f_1 = adj_mat * f_1
            f_2 = adj_mat * tf.transpose(f_2, [1,0])

            logits = tf.sparse_add(f_1, f_2)
            lrelu = tf.SparseTensor(indices=logits.indices, 
                    values=tf.nn.leaky_relu(logits.values), 
                    dense_shape=logits.dense_shape)
            coefs = tf.sparse_softmax(lrelu)

            if coef_drop != 0.0:
                coefs = tf.SparseTensor(indices=coefs.indices,
                        values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                        dense_shape=coefs.dense_shape)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

            # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
            # here we make an assumption that our input is of batch size 1, and reshape appropriately.
            # The method will fail in all other cases!
            coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
            seq_fts = tf.squeeze(seq_fts)
            vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
            vals = tf.expand_dims(vals, axis=0)
            vals.set_shape([1, nb_nodes, out_sz])
            ret = tf.contrib.layers.bias_add(vals)

            # residual connection
            if residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + tf.layers.dense(seq, ret.shape[-1]) # activation
                else:
                    ret = ret + seq

            return activation(ret)  # activation

    def inference(self, inputs, output_dim, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        h_1 = inputs 
        for i in range(0, len(hid_units)):
            attns = []
            for _ in range(n_heads[i]):
                attns.append(self.sp_attn_head(h_1,
                    adj_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation, nb_nodes=nb_nodes,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
    
        return h_1


