# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import socket

from reco_utils.recommender.deeprec.models.base_model_contrastive_noLong import BaseModel_Contrastive_noLong
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric, cal_weighted_metric, cal_mean_alpha_metric, load_dict

__all__ = ["SequentialBaseModel_Contrastive_noLong"]


class SequentialBaseModel_Contrastive_noLong(BaseModel_Contrastive_noLong):
    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        """Initializing the model. Create common logics which are needed by all sequential models, such as loss function, 
        parameter set.

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
            graph (obj): An optional graph.
            seed (int): Random seed.
        """
        self.hparams = hparams

        self.need_sample = hparams.need_sample
        self.train_num_ngs = hparams.train_num_ngs
        if self.train_num_ngs is None:
            raise ValueError(
                "Please confirm the number of negative samples for each positive instance."
            )
        self.min_seq_length = (
            hparams.min_seq_length if "min_seq_length" in hparams else 1
        )
        self.hidden_size = hparams.hidden_size if "hidden_size" in hparams else None
        self.graph = tf.Graph() if not graph else graph

        with self.graph.as_default():
            self.embedding_keeps = tf.placeholder(tf.float32, name="embedding_keeps")
            self.embedding_keep_prob_train = None
            self.embedding_keep_prob_test = None

        super().__init__(hparams, iterator_creator, graph=self.graph, seed=seed)

    @abc.abstractmethod
    def _build_seq_graph(self):
        """Subclass will implement this."""
        pass

    def _build_graph(self):
        """The main function to create sequential models.
        
        Returns:
            obj:the prediction score make by the model.
        """
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        self.embedding_keep_prob_train = 1.0 - hparams.embedding_dropout
        if hparams.test_dropout:
            self.embedding_keep_prob_test = 1.0 - hparams.embedding_dropout
        else:
            self.embedding_keep_prob_test = 1.0

        with tf.variable_scope("sequential") as self.sequential_scope:
            self._build_embedding()
            self._lookup_from_embedding()
            model_output_pos, loss_cosine = self._build_seq_graph()
            
            logit_pos = self._fcn_net(model_output_pos, hparams.layer_sizes, scope="logit_pos")
            # logit_neg = self._fcn_net(model_output_neg, hparams.layer_sizes, scope="logit_neg")

           # logitB = self._fcn_net(model_outputB, hparams.layer_sizes, scope="logit_fcnB")

            #self._add_norm()
            return logit_pos, loss_cosine

    def train(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_train
        return super(SequentialBaseModel_Contrastive_noLong, self).train(sess, feed_dict)


    #  def batch_train(self, file_iterator, train_sess, vm, tb):
    def batch_train(self, file_iterator, train_sess, vm, tb, valid_file, valid_num_ngs):
        """Train the model for a single epoch with mini-batches.

        Args:
            file_iterator (Iterator): iterator for training data.
            train_sess (Session): tf session for training.
            vm (VizManager): visualization manager for visdom.
            tb (TensorboardX): visualization manager for TensorboardX.

        Returns:
        epoch_loss: total loss of the single epoch.

        """
        step = 0
        epoch_loss = 0
        for batch_data_input in file_iterator:
           # if step == 0:
            #    print("domain A",list(batch_data_input.items())[4])
            if batch_data_input:
                # print(batch_data_input)
                # import pdb; pdb.set_trace()
                step_result = self.train(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, summary) = step_result
                # print(step_loss)

                #  (_, _, step_loss, step_data_loss, summary, _, _, _, _, _, _) = step_result
                #  (_, _, step_loss, step_data_loss, summary, _, _, _,) = step_result
                if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                    # print(summary)
                    # import pdb; pdb.set_trace()
                    self.writer.add_summary(summary, step)
                epoch_loss += step_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss, step_data_loss
                        )
                    )
                   # print("Gradient A", step_gradient)
                    if self.hparams.visual_type == 'epoch':
                        if vm != None:
                            vm.step_update_line('loss', step_loss)
                        #  tf.summary.scalar('loss',step_loss)
                        tb.add_scalar('loss', step_loss, step)
               # if step % 600 == 0:  break

                if self.hparams.visual_type == 'step':
                    if step % self.hparams.visual_step == 0:
                        if vm != None:
                            vm.step_update_line('loss', step_loss)
                        #  tf.summary.scalar('loss',step_loss)
                        tb.add_scalar('loss', step_loss, step)

                        # steps validation for visualization
                        valid_res = self.run_weighted_eval(valid_file, valid_num_ngs)
                        if vm != None:
                            vm.step_update_multi_lines(valid_res)  # TODO
                        for vs in valid_res:
                            #  tf.summary.scalar(vs.replace('@', '_'), valid_res[vs])
                            tb.add_scalar(vs.replace('@', '_'), valid_res[vs], step)


        return epoch_loss

    def fit(
        self, train_fileA, valid_fileA, valid_num_ngs, eval_metric="group_auc", vm=None, tb=None, pretrain=False
    ):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_file is not None, evaluate it too.
        
        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            valid_num_ngs (int): the number of negative instances with one positive instance in validation data.
            eval_metric (str): the metric that control early stopping. e.g. "auc", "group_auc", etc.

        Returns:
            obj: An instance of self.
        """

        # check bad input.
        if not self.need_sample and self.train_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for training without sampling needed."
            )
        if valid_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for validation."
            )

        if self.need_sample and self.train_num_ngs < 1:
            self.train_num_ngs = 1

        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            if not os.path.exists(self.hparams.SUMMARIES_DIR):
                os.makedirs(self.hparams.SUMMARIES_DIR)

            self.writer = tf.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        #  if pretrain:
            #  self.saver_emb = tf.train.Saver({'item_lookup':'item_embedding', 'user_lookup':'user_embedding'},max_to_keep=self.hparams.epochs)
        if pretrain:
            print('start saving embedding')
            if not os.path.exists(self.hparams.PRETRAIN_DIR):
                os.makedirs(self.hparams.PRETRAIN_DIR)
            #  checkpoint_emb_path = self.saver_emb.save(
                #  sess=train_sess,
                #  save_path=self.hparams.PRETRAIN_DIR + "epoch_" + str(epoch),
            #  )
            #  graph_def = tf.get_default_graph().as_graph_def()
            var_list = ['sequential/embedding/item_embedding', 'sequential/embedding/user_embedding']
            constant_graph = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, var_list)
            with tf.gfile.FastGFile(self.hparams.PRETRAIN_DIR + "test-model.pb", mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            print('embedding saved')

        train_sess = self.sess
        eval_info = list()

        best_metric_A, self.best_epoch = 0, 0

        for epoch in range(1, self.hparams.epochs + 1):
            self.hparams.current_epoch = epoch
            file_iteratorA = self.iterator.load_data_from_file(
                train_fileA,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=self.train_num_ngs,
            )
            #  epoch_loss = self.batch_train(file_iterator, train_sess, vm, tb)
            print("-------------------------------------------------------------------")
            # print("domain A:")

            epoch_lossA = self.batch_train(file_iteratorA, train_sess, vm, tb, valid_fileA, valid_num_ngs)
            # epoch_lossA = 0
            if vm != None:
                vm.step_update_line('epoch lossA', epoch_lossA)

            #  tf.summary.scalar('epoch loss', epoch_loss)
            tb.add_scalar('epoch_lossA', epoch_lossA, epoch)

            valid_resA = self.run_weighted_eval_valid(valid_fileA, valid_num_ngs)
            # valid_res_user = self.run_weighted_eval_userSeq(valid_fileA, valid_num_ngs)

            print(
                "eval valid at epoch {0}: {1},".format(
                    epoch,
                    ",".join(
                        [
                            "" + str(key) + ":" + str(value)
                            for key, value in valid_resA.items()
                        ]
                    ),
                )
            )
            if self.hparams.visual_type == 'epoch':
                if vm != None:
                    vm.step_update_multi_lines(valid_resA)  # TODO
                    vm.step_update_multi_lines(valid_resB)  # TODO

                for vs in valid_resA:
                    #  tf.summary.scalar(vs.replace('@', '_'), valid_res[vs])
                    tb.add_scalar(vs.replace('@', '_'), valid_resA[vs], epoch)

            eval_info.append((epoch, valid_resA))

            progress = False
            early_stop = self.hparams.EARLY_STOP
 
            if valid_resA[eval_metric] > best_metric_A:
                best_metric_A = valid_resA[eval_metric]
                self.best_epoch = epoch
                progress = True
            else:
                if early_stop > 0 and epoch - self.best_epoch >= early_stop:
                    print("early stop at epoch {0}!".format(epoch))
 
                    if pretrain:
                        if not os.path.exists(self.hparams.PRETRAIN_DIR):
                            os.makedirs(self.hparams.PRETRAIN_DIR)
                        #  checkpoint_emb_path = self.saver_emb.save(
                            #  sess=train_sess,
                            #  save_path=self.hparams.PRETRAIN_DIR + "epoch_" + str(epoch),
                        #  )
                        #  graph_def = tf.get_default_graph().as_graph_def()
                        var_list = ['sequential/embedding/item_embedding', 'sequential/embedding/user_embedding']
                        constant_graph = tf.graph_util.convert_variables_to_constants(train_sess, train_sess.graph_def, var_list)
                        with tf.gfile.FastGFile(self.hparams.PRETRAIN_DIR + "test-model.pb", mode='wb') as f:
                            f.write(constant_graph.SerializeToString())
 
                    break

            if self.hparams.save_model and self.hparams.MODEL_DIR:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                if progress:
                    checkpoint_path = self.saver.save(
                        sess=train_sess,
                        save_path=self.hparams.MODEL_DIR + "epoch_" + str(epoch),
                    )


        if self.hparams.write_tfevents:
            self.writer.close()

        print(eval_info)
        print("best epoch: {0}".format(self.best_epoch))
        return self

    def run_eval(self, filename, num_ngs):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                step_pred, step_labels = self.eval(load_sess, batch_data_input)
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        return res

    def eval(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        return super(SequentialBaseModel, self).eval(sess, feed_dict)

    def run_weighted_eval_neg(self, filename, num_ngs, calc_mean_alpha=False):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        items = []
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1
        if calc_mean_alpha:
            alphas = []

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                if not calc_mean_alpha:
                    step_item, step_pred, step_labels = self.eval_with_user(load_sess, batch_data_input)
                step_pred = 1 - step_pred
                items.extend(np.reshape(step_item, -1))
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                # print(step_pred.shape)
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        res_weighted = cal_weighted_metric(items, preds, labels, self.hparams.weighted_metrics)
        res.update(res_weighted)

        return res

    def run_weighted_eval_valid(self, filename, num_ngs, calc_mean_alpha=False):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        users = []
        preds = []
        labels = []
        current_preds = []
        current_labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1
        if calc_mean_alpha:
            alphas = []
        last_user = None

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                if not calc_mean_alpha:
                    step_user, step_pred, step_labels = self.eval_with_user(load_sess, batch_data_input)
                else:
                    step_user, step_pred, step_labels, step_alpha = self.eval_with_user_and_alpha(load_sess, batch_data_input)
                    alphas.extend(np.reshape(step_alpha, -1))
                users.extend(np.reshape(step_user, -1))
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))

                # group_preds.extend(np.reshape(step_pred, (-1, group)))
                # group_labels.extend(np.reshape(step_labels, (-1, group)))
        

        res = cal_metric(labels, preds, self.hparams.metrics)
        # res_pairwise = cal_metric(
        #     group_labels, group_preds, self.hparams.pairwise_metrics
        # )
        # res.update(res_pairwise)

        # res_weighted = cal_weighted_metric(users, preds, labels, self.hparams.weighted_metrics)
        # res.update(res_weighted)
        if calc_mean_alpha:
            res_alpha = cal_mean_alpha_metric(alphas, labels)
            res.update(res_alpha)
        return res

    def run_weighted_eval(self, filename, num_ngs, calc_mean_alpha=False):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        users = []
        preds = []
        labels = []
        current_preds = []
        current_labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1
        if calc_mean_alpha:
            alphas = []
        last_user = None

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                if not calc_mean_alpha:
                    step_user, step_pred, step_labels = self.eval_with_user(load_sess, batch_data_input)
                else:
                    step_user, step_pred, step_labels, step_alpha = self.eval_with_user_and_alpha(load_sess, batch_data_input)
                    alphas.extend(np.reshape(step_alpha, -1))
                users.extend(np.reshape(step_user, -1))
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                # print(step_attn_score)
                # print(step_relation)
                # print(step_pred.shape)

                # if (step_user == last_user):
                #     current_preds.append(preds)
                #     current_labels.append(preds)
                # if (len(step_labels) == 62):
                #     import pdb; pdb.set_trace()
                # print()
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))
        
        # df = pd.DataFrame({"users":users, "preds":preds, "labels":labels})
        # group_preds_df = df.groupby("users")['preds'].apply(list).reset_index(name='preds_g')['preds_g']
        # group_labels_df = df.groupby("users")['labels'].apply(list).reset_index(name='labels_g')['labels_g']
        # for i in group_preds_df:
        #     # if (len(i) > 2):
        #     group_preds.append(i)
        # for i in group_labels_df:
        #     # if (len(i) > 2):
        #     group_labels.append(i)
        # import pdb 
        # pdb.set_trace()

        # for g in list(group):
        #     group_preds.append(g["preds"])
        #     group_labels.append(g["labels"])
        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        # print(users, labels)
        # import pdb
        # pdb.set_trace()
        res_weighted = cal_weighted_metric(users, preds, labels, self.hparams.weighted_metrics)
        res.update(res_weighted)
        if calc_mean_alpha:
            res_alpha = cal_mean_alpha_metric(alphas, labels)
            res.update(res_alpha)
        return res

    def eval_with_user(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.users, self.predA, self.iterator.labels], feed_dict=feed_dict)

    def eval_with_item(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.items, self.predA, self.iterator.labels], feed_dict=feed_dict)

    def eval_with_user_and_alpha(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.users, self.pred, self.iterator.labels, self.alpha_output], feed_dict=feed_dict)

    def predict(self, infile_name, outfile_name):
        """Make predictions on the given data, and output predicted scores to a file.
        
        Args:
            infile_name (str): Input file name.
            outfile_name (str): Output file name.

        Returns:
            obj: An instance of self.
        """

        load_sess = self.sess
        with tf.gfile.GFile(outfile_name, "w") as wt:
            for batch_data_input in self.iterator.load_data_from_file(
                infile_name, batch_num_ngs=0
            ):
                if batch_data_input:
                    step_pred = self.infer(load_sess, batch_data_input)
                    step_pred = np.reshape(step_pred, -1)
                    wt.write("\n".join(map(str, step_pred)))
                    wt.write("\n")
        return self

    def infer(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        return super(SequentialBaseModel, self).infer(sess, feed_dict)

    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        hparams = self.hparams
        self.user_vocab_length = len(load_dict(hparams.user_vocab))
        self.item_vocab_length = len(load_dict(hparams.item_vocab))
        self.cate_vocab_length = len(load_dict(hparams.cate_vocab))
        self.user_embedding_dim = hparams.user_embedding_dim
        self.item_embedding_dim = hparams.item_embedding_dim
        self.cate_embedding_dim = hparams.cate_embedding_dim

        with tf.variable_scope("embedding", initializer=self.initializer):
            self.user_lookup = tf.get_variable(
                name="user_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.item_lookup = tf.get_variable(
                name="item_embedding",
                shape=[self.item_vocab_length, self.item_embedding_dim],
                dtype=tf.float32,
            )
            self.label_lookup = tf.get_variable(
                name="cate_embedding",
                shape=[self.cate_vocab_length, self.cate_embedding_dim],
                dtype=tf.float32,
            )
            self.position = tf.get_variable(
                name="position_embedding",
                shape=[hparams.max_seq_length, self.item_embedding_dim],
                dtype=tf.float32,
            )
        print(self.hparams.FINETUNE_DIR)
        print(not self.hparams.FINETUNE_DIR)
        if self.hparams.FINETUNE_DIR:
            # import pdb; pdb.set_trace()
            with tf.Session() as sess:
                # with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
                #     graph_def = tf.GraphDef()
                #     graph_def.ParseFromString(f.read())
                #     sess.graph.as_default()
                #     tf.import_graph_def(graph_def, name='')
                #  tf.global_variables_initializer().run()
                output_graph_def = tf.GraphDef()
                with open(self.hparams.FINETUNE_DIR + "test-model.pb", "rb") as f:
                    output_graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(output_graph_def, name="")

                self.item_lookup = sess.graph.get_tensor_by_name('sequential/embedding/item_embedding')
                self.user_lookup = sess.graph.get_tensor_by_name('sequential/embedding/user_embedding')
            #  print(input_x.eval())

            #  output = sess.graph.get_tensor_by_name("conv/b:0")

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        self.user_embedding = tf.nn.embedding_lookup( # iterator_users overlap?
            self.user_lookup, self.iterator.users
        )
        tf.summary.histogram("user_embedding_output", self.user_embedding)

        self.item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.items
        )
        self.user_history_embedding = tf.nn.embedding_lookup(
            self.user_lookup, self.iterator.user_history
        )
        self.item_history_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_history
        )
        tf.summary.histogram(
            "item_history_embedding_output", self.item_history_embedding
        )

        self.position_embedding = tf.nn.embedding_lookup(
            self.position, tf.tile(tf.expand_dims(tf.range(tf.shape(self.iterator.item_history)[1]), 0), [tf.shape(self.iterator.item_history)[0], 1]), # B*L 
        )
        tf.summary.histogram(
            "position_embedding_output", self.position_embedding
        )

        self.cate_embedding = tf.nn.embedding_lookup(
            self.label_lookup, self.iterator.cates
        )
        self.label_history_embedding = tf.nn.embedding_lookup(
            self.label_lookup, self.iterator.user_history
        )
        tf.summary.histogram(
            "cate_history_embedding_output", self.label_history_embedding
        )

        involved_items = tf.concat(
            [
                tf.reshape(self.iterator.item_history, [-1]),
                tf.reshape(self.iterator.items, [-1]),
            ],
            -1,
        )
        self.involved_items, _ = tf.unique(involved_items)
        involved_item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.involved_items
        )
        self.embed_params.append(involved_item_embedding)

        involved_cates = tf.concat(
            [
                tf.reshape(self.iterator.item_cate_history, [-1]),
                tf.reshape(self.iterator.cates, [-1]),
            ],
            -1,
        )
        self.involved_cates, _ = tf.unique(involved_cates)
        involved_cate_embedding = tf.nn.embedding_lookup(
            self.label_lookup, self.involved_cates
        )
        self.embed_params.append(involved_cate_embedding)

        self.target_item_embedding = self.item_embedding
        self.target_user_embedding = self.user_embedding
        tf.summary.histogram("target_item_embedding_output", self.target_item_embedding)

        # dropout after embedding
        self.user_embedding = self._dropout(
            self.user_embedding, keep_prob=self.embedding_keeps
        )
        self.item_history_embedding = self._dropout(
            self.item_history_embedding, keep_prob=self.embedding_keeps
        )
        self.position_embedding = self._dropout(
            self.position_embedding, keep_prob=self.embedding_keeps
        )

        self.label_history_embedding = self._dropout(
            self.label_history_embedding, keep_prob=self.embedding_keeps
        )
        self.target_item_embedding = self._dropout(
            self.target_item_embedding, keep_prob=self.embedding_keeps
        )
        self.target_user_embedding = self._dropout(
            self.target_user_embedding, keep_prob=self.embedding_keeps
        )
    def _add_norm(self):
        """Regularization for embedding variables and other variables."""
        all_variables, embed_variables = (
            tf.trainable_variables(),
            tf.trainable_variables(self.sequential_scope._name + "/embedding"),
        )
        layer_params = list(set(all_variables) - set(embed_variables))
        self.layer_params.extend(layer_params)
