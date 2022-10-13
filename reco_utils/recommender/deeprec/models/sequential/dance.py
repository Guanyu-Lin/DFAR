# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import (
    Time4LSTMCell,
)
from reco_utils.recommender.deeprec.deeprec_utils import load_dict

import os
import numpy as np

__all__ = ["DANCEModel"]


class DANCEModel(SequentialBaseModel):

    def _get_loss(self):
        """Make loss function, consists of data loss, regularization loss, contrastive loss and discrepancy loss
        
        Returns:
            obj: Loss value
        """
        self.data_loss = self._compute_data_loss()
        self.regular_loss = self._compute_regular_loss()
        self.contrastive_loss = self._compute_contrastive_loss()
        self.discrepancy_loss = self._compute_discrepancy_loss()

        self.pretrain_loss = self.data_loss + self.regular_loss + self.contrastive_loss + self.discrepancy_loss
        self.finetune_loss = self.data_loss + self.regular_loss
        self.loss = self.data_loss + self.regular_loss + self.contrastive_loss + self.discrepancy_loss
        return self.loss

    def _build_train_opt(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            obj: An operation that applies the specified optimization step.
        """
        self.pretrain_update = self._build_pretrain_opt()
        self.finetune_update = self._build_finetune_opt()

        return super(DANCEModel, self)._build_train_opt()

    def _compute_contrastive_loss(self):
        """Contrative loss on long and short term intention."""
        contrastive_mask = tf.where(
            tf.greater(self.sequence_length, self.hparams.contrastive_length_threshold),
            tf.ones_like(self.sequence_length, dtype=tf.float32),
            tf.zeros_like(self.sequence_length, dtype=tf.float32)
        )
        long_mean_recent_loss = tf.reduce_sum(contrastive_mask*tf.math.softplus(tf.reduce_sum(self.att_fea_long*(-self.hist_mean + self.hist_recent), -1)))/tf.reduce_sum(contrastive_mask)
        short_recent_mean_loss = tf.reduce_sum(contrastive_mask*tf.math.softplus(tf.reduce_sum(self.att_fea_short*(-self.hist_recent + self.hist_mean), -1)))/tf.reduce_sum(contrastive_mask)
        mean_long_short_loss = tf.reduce_sum(contrastive_mask*tf.math.softplus(tf.reduce_sum(self.hist_mean*(-self.att_fea_long + self.att_fea_short), -1)))/tf.reduce_sum(contrastive_mask)
        recent_short_long_loss = tf.reduce_sum(contrastive_mask*tf.math.softplus(tf.reduce_sum(self.hist_recent*(-self.att_fea_short + self.att_fea_long), -1)))/tf.reduce_sum(contrastive_mask)
        if self.hparams.manual_alpha and self.hparams.manual_alpha_value*(1 - self.hparams.manual_alpha_value) == 0:
            if self.hparams.manual_alpha_value == 1:
                contrastive_loss = long_mean_recent_loss
            else:
                contrastive_loss = short_recent_mean_loss
        else:
            contrastive_loss = long_mean_recent_loss + short_recent_mean_loss + mean_long_short_loss + recent_short_long_loss
        contrastive_loss = tf.multiply(self.hparams.contrastive_loss_weight, contrastive_loss)
        return contrastive_loss

    def _compute_discrepancy_loss(self):
        """Discrepancy loss between long and short term user embeddings."""
        discrepancy_loss = tf.reduce_mean(
            tf.math.squared_difference(
                tf.reshape(self.involved_user_long_embedding, [-1]),
                tf.reshape(self.involved_user_short_embedding, [-1])
            )
        )
        discrepancy_loss = -tf.multiply(self.hparams.discrepancy_loss_weight, discrepancy_loss)
        return discrepancy_loss

    def _build_pretrain_opt(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            obj: An operation that applies the specified optimization step.
        """
        train_step = self._train_opt()
        gradients, variables = zip(*train_step.compute_gradients(self.pretrain_loss))
        if self.hparams.is_clip_norm:
            gradients = [
                None
                if gradient is None
                else tf.clip_by_norm(gradient, self.hparams.max_grad_norm)
                for gradient in gradients
            ]
        return train_step.apply_gradients(zip(gradients, variables))

    def _build_finetune_opt(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            obj: An operation that applies the specified optimization step.
        """
        train_step = self._train_opt()
        gradients, variables = zip(*train_step.compute_gradients(self.finetune_loss))
        if self.hparams.is_clip_norm:
            gradients = [
                None
                if gradient is None
                else tf.clip_by_norm(gradient, self.hparams.max_grad_norm)
                for gradient in gradients
            ]
        return train_step.apply_gradients(zip(gradients, variables))

    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        super(DANCEModel, self)._build_embedding()
        hparams = self.hparams
        self.user_vocab_length = len(load_dict(hparams.user_vocab))
        self.user_embedding_dim = hparams.user_embedding_dim

        with tf.variable_scope("embedding", initializer=self.initializer):
            self.user_long_lookup = tf.get_variable(
                name="user_long_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.user_short_lookup = tf.get_variable(
                name="user_short_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        super(DANCEModel, self)._lookup_from_embedding()

        self.user_long_embedding = tf.nn.embedding_lookup(
            self.user_long_lookup, self.iterator.users
        )
        tf.summary.histogram("user_long_embedding_output", self.user_long_embedding)

        self.user_short_embedding = tf.nn.embedding_lookup(
            self.user_short_lookup, self.iterator.users
        )
        tf.summary.histogram("user_short_embedding_output", self.user_short_embedding)

        involved_users = tf.reshape(self.iterator.users, [-1])
        self.involved_users, _ = tf.unique(involved_users)
        self.involved_user_long_embedding = tf.nn.embedding_lookup(
            self.user_long_lookup, self.involved_users
        )
        self.embed_params.append(self.involved_user_long_embedding)
        self.involved_user_short_embedding = tf.nn.embedding_lookup(
            self.user_short_lookup, self.involved_users
        )
        self.embed_params.append(self.involved_user_short_embedding)

        # dropout after embedding
        self.user_long_embedding = self._dropout(
            self.user_long_embedding, keep_prob=self.embedding_keeps
        )
        self.user_short_embedding = self._dropout(
            self.user_short_embedding, keep_prob=self.embedding_keeps
        )

    def _build_seq_graph(self):
        """The main function to create dance model.
        
        Returns:
            obj:the output of dance section.
        """
        hparams = self.hparams
        with tf.variable_scope("dance"):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            self.mask = self.iterator.mask
            self.real_mask = tf.cast(self.mask, tf.float32)
            self.sequence_length = tf.reduce_sum(self.mask, 1)

            with tf.variable_scope("long_term"):
                if not hparams.use_complex_attention:
                    att_outputs_long = self._attention_transform_inner_product(self.user_long_embedding, hist_input)
                else:
                    att_outputs_long = self._attention_fcn(self.user_long_embedding, hist_input)
                self.att_fea_long = tf.reduce_sum(att_outputs_long, 1)
                tf.summary.histogram("att_fea_long", self.att_fea_long)

                self.hist_mean = tf.reduce_sum(hist_input*tf.expand_dims(self.real_mask, -1), 1)/tf.reduce_sum(self.real_mask, 1, keepdims=True)

            with tf.variable_scope("short_term"):
                if hparams.interest_evolve:
                    _, short_term_intention = dynamic_rnn(
                        tf.nn.rnn_cell.GRUCell(hparams.user_embedding_dim),
                        inputs=hist_input,
                        sequence_length=self.sequence_length,
                        initial_state=self.user_short_embedding,
                        dtype=tf.float32,
                        scope="short_term_intention",
                    )
                else:
                    short_term_intention = self.user_short_embedding
                tf.summary.histogram("GRU_final_state", short_term_intention)

                self.position = tf.math.cumsum(self.real_mask, axis=1, reverse=True)
                self.recent_mask = tf.logical_and(self.position >= 1, self.position <= hparams.contrastive_recent_k)
                self.real_recent_mask = tf.where(self.recent_mask, tf.ones_like(self.recent_mask, dtype=tf.float32), tf.zeros_like(self.recent_mask, dtype=tf.float32))

                self.hist_recent = tf.reduce_sum(hist_input*tf.expand_dims(self.real_recent_mask, -1), 1)/tf.reduce_sum(self.real_recent_mask, 1, keepdims=True)

                if hparams.use_time4lstm:
                    item_history_embedding_new = tf.concat(
                        [
                            hist_input,
                            tf.expand_dims(self.iterator.time_from_first_action, -1),
                        ],
                        -1,
                    )
                    item_history_embedding_new = tf.concat(
                        [
                            item_history_embedding_new,
                            tf.expand_dims(self.iterator.time_to_now, -1),
                        ],
                        -1,
                    )
                    rnn_outputs, _ = dynamic_rnn(
                        Time4LSTMCell(hparams.hidden_size),
                        inputs=item_history_embedding_new,
                        sequence_length=self.sequence_length,
                        dtype=tf.float32,
                        scope="time4lstm",
                    )
                else:
                    rnn_outputs, _ = dynamic_rnn(
                        tf.nn.rnn_cell.GRUCell(hparams.hidden_size),
                        inputs=hist_input,
                        sequence_length=self.sequence_length,
                        dtype=tf.float32,
                        scope="simple_gru",
                    )
                tf.summary.histogram("LSTM_outputs", rnn_outputs)

                short_term_query = tf.concat([short_term_intention, self.target_item_embedding], -1)
                if not hparams.use_complex_attention:
                    att_outputs_short = self._attention_transform_inner_product(short_term_query, rnn_outputs)
                else:
                    att_outputs_short = self._attention_fcn(short_term_query, rnn_outputs)
                self.att_fea_short = tf.reduce_sum(att_outputs_short, 1)
                tf.summary.histogram("att_fea2", self.att_fea_short)

            # ensemble
            with tf.name_scope("alpha"):

                if not hparams.manual_alpha:
                    if hparams.predict_long_short:
                        with tf.variable_scope("causal2"):
                            _, final_state = dynamic_rnn(
                                tf.nn.rnn_cell.GRUCell(hparams.hidden_size),
                                inputs=hist_input,
                                sequence_length=self.sequence_length,
                                dtype=tf.float32,
                                scope="causal2",
                            )
                            tf.summary.histogram("causal2", final_state)

                        concat_all = tf.concat(
                            [
                                final_state,
                                self.target_item_embedding,
                                self.att_fea_long,
                                self.att_fea_short,
                                tf.expand_dims(self.iterator.time_to_now[:, -1], -1),
                            ],
                            1,
                        )
                    else:
                        concat_all = tf.concat(
                            [
                                self.target_item_embedding,
                                self.att_fea_long,
                                self.att_fea_short,
                                tf.expand_dims(self.iterator.time_to_now[:, -1], -1),
                            ],
                            1,
                        )

                    last_hidden_nn_layer = concat_all
                    if hparams.vector_alpha:
                        fcn_transform_layer_sizes = hparams.att_fcn_layer_sizes[:-1] + [self.att_fea_long.shape[-1]]
                        alpha_logit = self._fcn_transform_net(
                            last_hidden_nn_layer, fcn_transform_layer_sizes, scope="fcn_alpha"
                        )
                    else:
                        alpha_logit = self._fcn_net(
                            last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="fcn_alpha"
                        )
                    self.alpha_output = tf.sigmoid(alpha_logit)
                    user_embed = self.att_fea_long * self.alpha_output + self.att_fea_short * (1.0 - self.alpha_output)
                    tf.summary.histogram("alpha", self.alpha_output)
                    if hparams.vector_alpha:
                        self.alpha_output_mean = tf.reduce_mean(self.alpha_output, -1)
                    else:
                        self.alpha_output_mean = self.alpha_output
                    error_with_category = self.alpha_output_mean - self.iterator.attn_labels
                    tf.summary.histogram("error_with_category", error_with_category)
                    squared_error_with_category = tf.math.sqrt(tf.math.squared_difference(tf.reshape(self.alpha_output_mean, [-1]), tf.reshape(self.iterator.attn_labels, [-1])))
                    tf.summary.histogram("squared_error_with_category", squared_error_with_category)
                else:
                    user_embed = self.att_fea_long * hparams.manual_alpha_value + self.att_fea_short * (1.0 - hparams.manual_alpha_value)
            model_output = tf.concat([user_embed, self.target_item_embedding], 1)
            tf.summary.histogram("model_output", model_output)
            return model_output

    def _fcn_transform_net(self, model_output, layer_sizes, scope):
        """Construct the MLP part for the model.

        Args:
            model_output (obj): The output of upper layers, input of MLP part
            layer_sizes (list): The shape of each layer of MLP part
            scope (obj): The scope of MLP part

        Returns:s
            obj: prediction logit after fully connected layer
        """
        hparams = self.hparams
        with tf.variable_scope(scope):
            last_layer_size = model_output.shape[-1]
            layer_idx = 0
            hidden_nn_layers = []
            hidden_nn_layers.append(model_output)
            with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
                for idx, layer_size in enumerate(layer_sizes):
                    curr_w_nn_layer = tf.get_variable(
                        name="w_nn_layer" + str(layer_idx),
                        shape=[last_layer_size, layer_size],
                        dtype=tf.float32,
                    )
                    curr_b_nn_layer = tf.get_variable(
                        name="b_nn_layer" + str(layer_idx),
                        shape=[layer_size],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                    )
                    tf.summary.histogram(
                        "nn_part/" + "w_nn_layer" + str(layer_idx), curr_w_nn_layer
                    )
                    tf.summary.histogram(
                        "nn_part/" + "b_nn_layer" + str(layer_idx), curr_b_nn_layer
                    )
                    curr_hidden_nn_layer = (
                        tf.tensordot(
                            hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                        )
                        + curr_b_nn_layer
                    )

                    scope = "nn_part" + str(idx)
                    activation = hparams.activation[idx]

                    if hparams.enable_BN is True:
                        curr_hidden_nn_layer = tf.layers.batch_normalization(
                            curr_hidden_nn_layer,
                            momentum=0.95,
                            epsilon=0.0001,
                            training=self.is_train_stage,
                        )

                    curr_hidden_nn_layer = self._active_layer(
                        logit=curr_hidden_nn_layer, activation=activation, layer_idx=idx
                    )
                    hidden_nn_layers.append(curr_hidden_nn_layer)
                    layer_idx += 1
                    last_layer_size = layer_size

                nn_output = hidden_nn_layers[-1]
                return nn_output

    def _attention_fcn(self, query, user_embedding):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item which is regarded as a query in attention operations.
            user_embedding (obj): The output of RNN layers which is regarded as user modeling.

        Returns:
            obj: Weighted sum of user modeling.
        """
        hparams = self.hparams
        with tf.variable_scope("attention_fcn"):
            query_size = query.shape[1].value
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            attention_mat = tf.get_variable(
                name="attention_mat",
                shape=[user_embedding.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(user_embedding, attention_mat, [[2], [0]])

            queries = tf.reshape(
                tf.tile(query, [1, att_inputs.shape[1].value]), tf.shape(att_inputs)
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
            return output

    def _attention_inner_product(self, query, user_embedding):
        """Apply attention by simple inner product.

        Args:
            query (obj): The embedding of target item which is regarded as a query in attention operations.
            user_embedding (obj): The output of RNN layers which is regarded as user modeling.

        Returns:
            obj: Weighted sum of user modeling.
        """
        with tf.variable_scope("attention_fcn"):
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            att_output = tf.reduce_sum(user_embedding*tf.expand_dims(query, 1), -1)
            mask_paddings = tf.ones_like(att_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_output, mask_paddings),
                name="att_weights",
            )
            output = user_embedding * tf.expand_dims(att_weights, -1)
            return output

    def _attention_transform_inner_product(self, query, user_embedding):
        """Apply attention by simple inner product.

        Args:
            query (obj): The embedding of target item which is regarded as a query in attention operations.
            user_embedding (obj): The output of RNN layers which is regarded as user modeling.

        Returns:
            obj: Weighted sum of user modeling.
        """
        with tf.variable_scope("attention_fcn"):
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            query_size = query.shape[1].value
            attention_mat = tf.get_variable(
                name="attention_mat",
                shape=[user_embedding.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(user_embedding, attention_mat, [[2], [0]])

            att_output = tf.reduce_sum(att_inputs*tf.expand_dims(query, 1), -1)
            mask_paddings = tf.ones_like(att_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_output, mask_paddings),
                name="att_weights",
            )
            output = user_embedding * tf.expand_dims(att_weights, -1)
            return output

    def pretrain(self, sess, feed_dict):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_train
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_train
        feed_dict[self.is_train_stage] = True
        return sess.run(
            [
                self.pretrain_update,
                self.extra_update_ops,
                self.pretrain_loss,
                self.regular_loss,
                self.contrastive_loss,
                self.discrepancy_loss,
                self.merged,
            ],
            feed_dict=feed_dict,
        )

    def finetune(self, sess, feed_dict):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_train
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_train
        feed_dict[self.is_train_stage] = True
        return sess.run(
            [
                self.finetune_update,
                self.extra_update_ops,
                self.finetune_loss,
                self.data_loss,
                self.regular_loss,
                self.merged,
            ],
            feed_dict=feed_dict,
        )

    def train(self, sess, feed_dict):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_train
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_train
        feed_dict[self.is_train_stage] = True
        return sess.run(
            [
                self.update,
                self.extra_update_ops,
                self.loss,
                self.data_loss,
                self.regular_loss,
                self.contrastive_loss,
                self.discrepancy_loss,
                self.merged,
            ],
            feed_dict=feed_dict,
        )

    def fit(
        self, train_file, valid_file, valid_num_ngs, eval_metric="group_auc", vm=None, pretrain=False
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

        if not pretrain:
            return super(DANCEModel, self).fit(train_file, valid_file, valid_num_ngs, eval_metric, vm)
        else:
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

            pretrain_sess = self.sess
            pretrain_eval_info = list()

            pretrain_best_metric, self.pretrain_best_epoch = 0, 0

            for epoch in range(1, self.hparams.pretrain_epochs + 1):
                self.hparams.current_epoch = epoch
                file_iterator = self.iterator.load_data_from_file(
                    train_file,
                    min_seq_length=self.min_seq_length,
                    batch_num_ngs=self.train_num_ngs,
                )

                epoch_loss = self.pretrain_batch_train(file_iterator, pretrain_sess, vm)
                vm.step_update_line('pretrain epoch loss', epoch_loss)

                valid_res = self.run_eval(valid_file, valid_num_ngs)
                print(
                    "pretrain eval valid at epoch {0}: {1}".format(
                        epoch,
                        ",".join(
                            [
                                "" + str(key) + ":" + str(value)
                                for key, value in valid_res.items()
                            ]
                        ),
                    )
                )
                pretrain_valid_res = {'pretrain '+k: v for k, v in valid_res.items()}
                vm.step_update_multi_lines(pretrain_valid_res)
                pretrain_eval_info.append((epoch, pretrain_valid_res))

                progress = False
                early_stop = self.hparams.EARLY_STOP
                if valid_res[eval_metric] > pretrain_best_metric:
                    pretrain_best_metric = valid_res[eval_metric]
                    self.pretrain_best_epoch = epoch
                    progress = True
                else:
                    if early_stop > 0 and epoch - self.pretrain_best_epoch >= early_stop:
                        print("pretrain early stop at epoch {0}!".format(epoch))
                        break

                if self.hparams.save_model and self.hparams.MODEL_DIR:
                    if not os.path.exists(self.hparams.MODEL_DIR):
                        os.makedirs(self.hparams.MODEL_DIR)
                    if progress:
                        checkpoint_path = self.saver.save(
                            sess=pretrain_sess,
                            save_path=self.hparams.MODEL_DIR + "pretrain_epoch_" + str(epoch),
                        )

            print(pretrain_eval_info)


            finetune_sess = self.sess
            finetune_eval_info = list()

            finetune_best_metric, self.finetune_best_epoch = 0, 0

            for epoch in range(1, self.hparams.finetune_epochs + 1):
                self.hparams.current_epoch = epoch
                file_iterator = self.iterator.load_data_from_file(
                    train_file,
                    min_seq_length=self.min_seq_length,
                    batch_num_ngs=self.train_num_ngs,
                )

                epoch_loss = self.finetune_batch_train(file_iterator, finetune_sess, vm)
                vm.step_update_line('finetune epoch loss', epoch_loss)

                valid_res = self.run_eval(valid_file, valid_num_ngs)
                print(
                    "finetune eval valid at epoch {0}: {1}".format(
                        epoch,
                        ",".join(
                            [
                                "" + str(key) + ":" + str(value)
                                for key, value in valid_res.items()
                            ]
                        ),
                    )
                )
                finetune_valid_res = {'finetune '+k: v for k, v in valid_res.items()}
                vm.step_update_multi_lines(finetune_valid_res)
                finetune_eval_info.append((epoch, finetune_valid_res))

                progress = False
                early_stop = self.hparams.EARLY_STOP
                if valid_res[eval_metric] > finetune_best_metric:
                    finetune_best_metric = valid_res[eval_metric]
                    self.finetune_best_epoch = epoch
                    progress = True
                else:
                    if early_stop > 0 and epoch - self.finetune_best_epoch >= early_stop:
                        print("finetune early stop at epoch {0}!".format(epoch))
                        break

                if self.hparams.save_model and self.hparams.MODEL_DIR:
                    if not os.path.exists(self.hparams.MODEL_DIR):
                        os.makedirs(self.hparams.MODEL_DIR)
                    if progress:
                        checkpoint_path = self.saver.save(
                            sess=finetune_sess,
                            save_path=self.hparams.MODEL_DIR + "finetune_epoch_" + str(epoch),
                        )

            if self.hparams.write_tfevents:
                self.writer.close()

            print(finetune_eval_info)
            print("best epoch: {0}".format(self.finetune_best_epoch))
            return self

    def pretrain_batch_train(self, file_iterator, train_sess, vm):
        """Pretrain the model for a single epoch with mini-batches.

        Args:
            file_iterator (Iterator): iterator for training data.
            train_sess (Session): tf session for training.
            vm (VizManager): visualization manager for visdom.

        Returns:
        epoch_loss: total loss of the single epoch.

        """
        step = 0
        epoch_loss = 0
        epoch_regular_loss = 0
        epoch_contrastive_loss = 0
        epoch_discrepancy_loss = 0
        for batch_data_input in file_iterator:
            if batch_data_input:
                step_result = self.pretrain(train_sess, batch_data_input)
                (_, _, step_loss, step_regular_loss, step_contrastive_loss, step_discrepancy_loss, summary) = step_result
                if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                    self.writer.add_summary(summary, step)
                epoch_loss += step_loss
                epoch_regular_loss += step_regular_loss
                epoch_contrastive_loss += step_contrastive_loss
                epoch_discrepancy_loss += step_discrepancy_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, contrastive_loss: {2:.4f}".format(
                            step, step_loss, step_contrastive_loss
                        )
                    )
                    vm.step_update_line('pretrain loss', step_loss)
                    vm.step_update_line('pretrain regular_loss', step_regular_loss)
                    vm.step_update_line('pretrain contrastive_loss', step_contrastive_loss)
                    vm.step_update_line('pretrain discrepancy_loss', step_discrepancy_loss)

        vm.step_update_line('pretrain_epoch_regular_loss', epoch_regular_loss)
        vm.step_update_line('pretrain_epoch_contrastive_loss', epoch_contrastive_loss)
        vm.step_update_line('pretrain_epoch_discrepancy_loss', epoch_discrepancy_loss)

        return epoch_loss

    def finetune_batch_train(self, file_iterator, train_sess, vm):
        """Train the model for a single epoch with mini-batches.

        Args:
            file_iterator (Iterator): iterator for training data.
            train_sess (Session): tf session for training.
            vm (VizManager): visualization manager for visdom.

        Returns:
        epoch_loss: total loss of the single epoch.

        """
        step = 0
        epoch_loss = 0
        epoch_data_loss = 0
        epoch_regular_loss = 0
        for batch_data_input in file_iterator:
            if batch_data_input:
                step_result = self.finetune(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, step_regular_loss, summary) = step_result
                if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                    self.writer.add_summary(summary, step)
                epoch_loss += step_loss
                epoch_data_loss += step_data_loss
                epoch_regular_loss += step_regular_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss, step_data_loss
                        )
                    )
                    vm.step_update_line('loss', step_loss)
                    vm.step_update_line('data_loss', step_data_loss)
                    vm.step_update_line('regular_loss', step_regular_loss)

        vm.step_update_line('epoch_data_loss', epoch_data_loss)
        vm.step_update_line('epoch_regular_loss', epoch_regular_loss)

        return epoch_loss

    def batch_train(self, file_iterator, train_sess, vm):
        """Train the model for a single epoch with mini-batches.

        Args:
            file_iterator (Iterator): iterator for training data.
            train_sess (Session): tf session for training.
            vm (VizManager): visualization manager for visdom.

        Returns:
        epoch_loss: total loss of the single epoch.

        """
        step = 0
        epoch_loss = 0
        epoch_data_loss = 0
        epoch_regular_loss = 0
        epoch_contrastive_loss = 0
        epoch_discrepancy_loss = 0
        for batch_data_input in file_iterator:
            if batch_data_input:
                step_result = self.train(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, step_regular_loss, step_contrastive_loss, step_discrepancy_loss, summary) = step_result
                if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                    self.writer.add_summary(summary, step)
                epoch_loss += step_loss
                epoch_data_loss += step_data_loss
                epoch_regular_loss += step_regular_loss
                epoch_contrastive_loss += step_contrastive_loss
                epoch_discrepancy_loss += step_discrepancy_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss, step_data_loss
                        )
                    )
                    if vm != None:
                        vm.step_update_line('loss', step_loss)
                        vm.step_update_line('data_loss', step_data_loss)
                        vm.step_update_line('regular_loss', step_regular_loss)
                        vm.step_update_line('contrastive_loss', step_contrastive_loss)
                        vm.step_update_line('discrepancy_loss', step_discrepancy_loss)

        if vm != None:
            vm.step_update_line('epoch_data_loss', epoch_data_loss)
            vm.step_update_line('epoch_regular_loss', epoch_regular_loss)
            vm.step_update_line('epoch_contrastive_loss', epoch_contrastive_loss)
            vm.step_update_line('epoch_discrepancy_loss', epoch_discrepancy_loss)

        return epoch_loss

    def _add_summaries(self):
        tf.summary.scalar("data_loss", self.data_loss)
        tf.summary.scalar("regular_loss", self.regular_loss)
        tf.summary.scalar("contrastive_loss", self.contrastive_loss)
        tf.summary.scalar("discrepancy_loss", self.discrepancy_loss)
        tf.summary.scalar("loss", self.loss)
        merged = tf.summary.merge_all()
        return merged
