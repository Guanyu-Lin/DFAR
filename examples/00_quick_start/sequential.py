#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags
from absl import logging

import sys
sys.path.append("../../")
import os
# import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import socket
import getpass
import smtplib
from email.mime.text import MIMEText

import setproctitle

import tensorflow as tf
import time
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from reco_utils.common.constants import SEED
from reco_utils.recommender.deeprec.deeprec_utils import (
    prepare_hparams
)
from reco_utils.dataset.sequential_reviews import data_preprocessing
from reco_utils.dataset.sequential_reviews import group_sequence
from reco_utils.recommender.deeprec.models.sequential.sli_rec import SLI_RECModel

from reco_utils.recommender.deeprec.models.sequential.dance import DANCEModel
from reco_utils.recommender.deeprec.models.sequential.sasli_rec import SASLI_RECModel
from reco_utils.recommender.deeprec.models.sequential.asvd import A2SVDModel
from reco_utils.recommender.deeprec.models.sequential.caser import CaserModel
from reco_utils.recommender.deeprec.models.sequential.gru4rec import GRU4RecModel
from reco_utils.recommender.deeprec.models.sequential.din import DINModel
from reco_utils.recommender.deeprec.models.sequential.dien import DIENModel
from reco_utils.recommender.deeprec.models.sequential.gcn import SURGEModel
from reco_utils.recommender.deeprec.models.sequential.sasrec import SASRecModel
from reco_utils.recommender.deeprec.models.sequential.sasrec_factor import SASRecFactorizationModel

from reco_utils.recommender.deeprec.models.sequential.sasrecneg import SASRecNegModel
from reco_utils.recommender.deeprec.models.sequential.feedrec import SASRecFeedModel
from reco_utils.recommender.deeprec.models.sequential.dfn import DFN

from reco_utils.recommender.deeprec.models.sequential.sasrec_talking import SASRecTalkingModel


from reco_utils.recommender.deeprec.models.sequential.feedrecdual_talking import SASRecFeedDualTalkModel

from reco_utils.recommender.deeprec.models.sequential.feedrecdual_mask import SASRecFeedDualModelMASK
from reco_utils.recommender.deeprec.models.sequential.feedrecdual_factor import FeedDualModelFactor
from reco_utils.recommender.deeprec.models.sequential.feedrecdual_mha import FeedDualModelMHA

from reco_utils.recommender.deeprec.models.sequential.sasrecneg_bpr import SASRecNegBPRModel


from reco_utils.recommender.deeprec.models.sequential.informer import InformerModel
from reco_utils.recommender.deeprec.models.sequential.performer import PerformerModel
from reco_utils.recommender.deeprec.models.sequential.bst import BSTModel

from reco_utils.recommender.deeprec.io.sequential_iterator import SequentialIterator


from reco_utils.common.visdom_utils import VizManager
from tensorboardX import SummaryWriter
from tensorflow.python.tools import inspect_checkpoint as chkp

FLAGS = flags.FLAGS
if True:

    flags.DEFINE_string('name', 'kuaishou-SASRec', 'Experiment name.')

    flags.DEFINE_string('dataset', 'toys', 'Dataset name.')

    flags.DEFINE_integer('val_num_ngs', 9, 'Number of negative instances with a positiver instance for validation.')
    flags.DEFINE_integer('test_num_ngs', 9, 'Number of negative instances with a positive instance for testing.')
    flags.DEFINE_integer('batch_size', 200, 'Batch size.')

    flags.DEFINE_string('model', 'SASRec', 'Model name.')
    
    flags.DEFINE_string('gpu_id', '1', 'GPU ID.')
    flags.DEFINE_float('loss_l2', 1e-5, 'L2 regulation for loss.')

    flags.DEFINE_float('embed_l2', 1e-6, 'L2 regulation for embeddings.')
    flags.DEFINE_float('layer_l2', 1e-6, 'L2 regulation for layers.')
    flags.DEFINE_integer('contrastive_length_threshold', 5, 'Minimum sequence length value to apply contrastive loss.')
    flags.DEFINE_integer('contrastive_recent_k', 3, 'Use the most recent k embeddings to compute short-term proxy.')

flags.DEFINE_boolean('amp_time_unit', True, 'Whether to amplify unit for time stamp.')
flags.DEFINE_boolean('is_preprocess', False, 'Whether to amplify unit for time stamp.')
flags.DEFINE_boolean('is_onlyPos', False, 'Whether to amplify unit for time stamp.')

flags.DEFINE_boolean('only_test', False, 'Only test and do not train.')
flags.DEFINE_boolean('only_visualize', False, 'Only test and do not train.')
flags.DEFINE_integer('hidden_size', 32, 'hidden size.')

flags.DEFINE_boolean('test_dropout', False, 'Whether to dropout during evaluation.')
flags.DEFINE_boolean('write_prediction_to_file', False, 'Whether to write prediction to file.')
flags.DEFINE_boolean('test_counterfactual', False, 'Whether to test with counterfactual data.')
flags.DEFINE_string('test_counterfactual_mode', 'shuffle', 'Mode for counterfactual evaluation, could be original, shuffle or recent.')
flags.DEFINE_integer('counterfactual_recent_k', 10, 'Use recent k interactions to predict the target item.')
flags.DEFINE_boolean('pretrain', False, 'Whether to use pretrain and finetune.')
#  flags.DEFINE_boolean('finetune', True, 'Whether to use pretrain and finetune.')
#  flags.DEFINE_string('finetune_path', '/data/changjianxin/ls-recommenders/saves/GCN/gat-uii_last_pretrain/pretrain/', 'Save path.')
flags.DEFINE_string('finetune_path', '', 'Save path.')
flags.DEFINE_string('loss', 'log_loss', 'loss ')

flags.DEFINE_boolean('vector_alpha', False, 'Whether to use vector alpha for long short term fusion.')
flags.DEFINE_boolean('manual_alpha', False, 'Whether to use predefined alpha for long short term fusion.')
flags.DEFINE_float('manual_alpha_value', 0.5, 'Predifined alpha value for long short term fusion.')
flags.DEFINE_boolean('interest_evolve', True, 'Whether to use a GRU to model interest evolution.')
flags.DEFINE_boolean('predict_long_short', True, 'Predict whether the next interaction is driven by long-term interest or short-term interest.')
flags.DEFINE_enum('single_part', 'no', ['no', 'long', 'short'], 'Whether to use only long, only short or both.')
flags.DEFINE_integer('is_clip_norm', 1, 'Whether to clip gradient norm.')
flags.DEFINE_boolean('use_complex_attention', True, 'Whether to use complex attention like DIN.')
flags.DEFINE_boolean('use_time4lstm', True, 'Whether to use Time4LSTMCell proposed by SLIREC.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs.')
flags.DEFINE_integer('early_stop', 2, 'Patience for early stop.')
flags.DEFINE_integer('pretrain_epochs', 10, 'Number of pretrain epochs.')
flags.DEFINE_integer('finetune_epochs', 100, 'Number of finetune epochs.')
flags.DEFINE_string('data_path', os.path.join("..","..", "..", ".."), 'Data file path.')
if True:

    flags.DEFINE_string('save_path', '../../saves/', 'Save path.')
    #  flags.DEFINE_string('save_path', '../../saves_step/', 'Save path.')
flags.DEFINE_integer('train_num_ngs', 9, 'Number of negative instances with a positive instance for training.')
flags.DEFINE_float('sample_rate', 1.0, 'Fraction of samples for training and testing.')
flags.DEFINE_float('attn_loss_weight', 0.001, 'Loss weight for supervised attention.')
flags.DEFINE_float('discrepancy_loss_weight', 0.01, 'Loss weight for discrepancy between long and short term user embedding.')
flags.DEFINE_float('contrastive_loss_weight', 0.1, 'Loss weight for contrastive of long and short intention.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_integer('show_step', 50, 'Step for showing metrics.')
flags.DEFINE_integer('max_seq_length', 20, 'Step for showing metrics.')
flags.DEFINE_integer('num_heads', 2, 'head for attention.')

flags.DEFINE_string('visual_type', 'epoch', '') #  for epoch visual
#  flags.DEFINE_string('visual_type', 'step', 'Step for drawing metrics.') #  for step visual
flags.DEFINE_integer('visual_step', 50, 'Step for drawing metrics.')
flags.DEFINE_string('no_visual_host', 'kwai', '')
flags.DEFINE_string('components', '1,1,1,1', 'for ablation study')

flags.DEFINE_boolean('enable_mail_service', False, 'Whether to e-mail yourself after each run.')


def get_model(flags_obj, model_path, summary_path, pretrain_path, finetune_path, user_vocab, item_vocab, cate_vocab, train_num_ngs, data_path):

    EPOCHS = flags_obj.epochs
    BATCH_SIZE = flags_obj.batch_size
    RANDOM_SEED = None  # Set None for non-deterministic result

    flags_obj.amp_time_unit = flags_obj.amp_time_unit if flags_obj.model == 'DANCE' else False

    pairwise_metrics = ['mean_mrr', 'ndcg@2;4;6;8;10']
    weighted_metrics = ['wauc']
    max_seq_length = flags_obj.max_seq_length
    time_unit = 's' 
    HIDDEN_SIZE = flags_obj.hidden_size

      
    input_creator = SequentialIterator
 

    if flags_obj.single_part != 'no':
        flags_obj.manual_alpha = True
        if flags_obj.single_part == 'long':
            flags_obj.manual_alpha_value = 1.0
        else:
            flags_obj.manual_alpha_value = 0.0
    elif flags_obj.manual_alpha:
        if flags_obj.manual_alpha_value == 1.0:
            flags_obj.single_part = 'long'
        elif flags_obj.manual_alpha_value == 0.0:
            flags_obj.single_part = 'short'

    #SliRec
    if flags_obj.model == 'SLIREC':
        yaml_file = '../../reco_utils/recommender/deeprec/config/sli_rec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                manual_alpha=flags_obj.manual_alpha,
                                manual_alpha_value=flags_obj.manual_alpha_value,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                counterfactual_recent_k=flags_obj.counterfactual_recent_k,
                                test_dropout=flags_obj.test_dropout,
                                time_unit=time_unit,
                    )
        model = SLI_RECModel(hparams, input_creator, seed=RANDOM_SEED)
    elif flags_obj.model == 'DANCE':
        if flags_obj.single_part != 'no':
            flags_obj.discrepancy_loss_weight = 0.0
        yaml_file = '../../reco_utils/recommender/deeprec/config/dance.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                discrepancy_loss_weight=flags_obj.discrepancy_loss_weight,
                                contrastive_loss_weight=flags_obj.contrastive_loss_weight,
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                pretrain_epochs=flags_obj.pretrain_epochs,
                                finetune_epochs=flags_obj.finetune_epochs,
                                vector_alpha=flags_obj.vector_alpha,
                                manual_alpha=flags_obj.manual_alpha,
                                manual_alpha_value=flags_obj.manual_alpha_value,
                                interest_evolve=flags_obj.interest_evolve,
                                predict_long_short=flags_obj.predict_long_short,
                                is_clip_norm=flags_obj.is_clip_norm,
                                contrastive_length_threshold=flags_obj.contrastive_length_threshold,
                                contrastive_recent_k=flags_obj.contrastive_recent_k,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                counterfactual_recent_k=flags_obj.counterfactual_recent_k,
                                test_dropout=flags_obj.test_dropout,
                                use_complex_attention=flags_obj.use_complex_attention,
                                use_time4lstm=flags_obj.use_time4lstm,
                                time_unit=time_unit,
                    )
        model = DANCEModel(hparams, input_creator, seed=RANDOM_SEED)

    #SliRec
    elif flags_obj.model == 'SASLIREC':
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasli_rec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                attn_loss_weight=flags_obj.attn_loss_weight,
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                test_dropout=flags_obj.test_dropout,
                    )
        model = SASLI_RECModel(hparams, input_creator, seed=RANDOM_SEED)

    #GRU4REC
    elif flags_obj.model == 'GRU4REC':
        yaml_file = '../../reco_utils/recommender/deeprec/config/gru4rec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length,
                                hidden_size=HIDDEN_SIZE,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                test_dropout=flags_obj.test_dropout,
                    )
        model = GRU4RecModel(hparams, input_creator, seed=RANDOM_SEED)
    elif flags_obj.model == 'DUALGRU4REC':
        yaml_file = '../../reco_utils/recommender/deeprec/config/gru4rec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length,
                                hidden_size=HIDDEN_SIZE,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                test_dropout=flags_obj.test_dropout,
                    )
        model = DualGRU4RecModel(hparams, input_creator, seed=RANDOM_SEED)
    #DIN
    elif flags_obj.model == 'DIN':
        yaml_file = '../../reco_utils/recommender/deeprec/config/din.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, hidden_size=HIDDEN_SIZE,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                test_dropout=flags_obj.test_dropout,
                    )
        model = DINModel(hparams, input_creator, seed=RANDOM_SEED)

    # SURGE
    elif flags_obj.model == 'SURGE':
        yaml_file = '../../reco_utils/recommender/deeprec/config/gcn.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SURGEModel(hparams, input_creator, seed=RANDOM_SEED)
    # SASRec
    elif flags_obj.model == 'SASRecFactor':
        print("I am SASRecFactor")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SASRecFactorizationModel(hparams, input_creator, seed=RANDOM_SEED)
    # SASRec
    elif flags_obj.model == 'SASRec':
        print("I am SASRec")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SASRecModel(hparams, input_creator, seed=RANDOM_SEED)
    elif flags_obj.model == 'SASRecTalking':
        print("I am SASRec")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SASRecTalkingModel(hparams, input_creator, seed=RANDOM_SEED)

    elif flags_obj.model == 'SASRecNoLong':
        print("I am SASRec")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                components=flags_obj.components,
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SASRecFeedDualNoLongModel(hparams, input_creator, seed=RANDOM_SEED)
    elif flags_obj.model == 'DFN':
        print("I am DFN")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                components=flags_obj.components,
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = DFN(hparams, input_creator, seed=RANDOM_SEED)
    elif flags_obj.model == 'SASRecFeedDual':
        print("I am SASRec")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                components=flags_obj.components,
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SASRecFeedDualModel(hparams, input_creator, seed=RANDOM_SEED)
    elif flags_obj.model == 'feedDualMHA':
        print("I am feedDualMHA")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                loss=flags_obj.loss,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                num_heads=flags_obj.num_heads,
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = FeedDualModelMHA(hparams, input_creator, seed=RANDOM_SEED)
    elif flags_obj.model == 'DFAR':
        print("I am DFAR")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                loss=flags_obj.loss,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                num_heads=flags_obj.num_heads,
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = FeedDualModelFactor(hparams, input_creator, seed=RANDOM_SEED)
    elif flags_obj.model == 'SASRecFeedDualMask':
        print("I am SASRec")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                loss=flags_obj.loss,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SASRecFeedDualModelMASK(hparams, input_creator, seed=RANDOM_SEED)
    elif flags_obj.model == 'SASRecFeedDualTalk':
        print("I am SASRec")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SASRecFeedDualTalkModel(hparams, input_creator, seed=RANDOM_SEED)

    # SASRec
    elif flags_obj.model == 'SASRecFeed':
        print("I am SASRec")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SASRecFeedModel(hparams, input_creator, seed=RANDOM_SEED)

    elif flags_obj.model == 'SASRecNegBPR':
        print("I am SASRec")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SASRecNegBPRModel(hparams, input_creator, seed=RANDOM_SEED)
    elif flags_obj.model == 'SASRecNeg':
        print("I am SASRec")
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SASRecNegModel(hparams, input_creator, seed=RANDOM_SEED)

    # Informer
    elif flags_obj.model == 'Informer':
        yaml_file = '../../reco_utils/recommender/deeprec/config/informer.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = InformerModel(hparams, input_creator, seed=RANDOM_SEED)

    # Performer
    elif flags_obj.model == 'Performer':
        print("I am Performer")
        yaml_file = '../../reco_utils/recommender/deeprec/config/performer.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = PerformerModel(hparams, input_creator, seed=RANDOM_SEED)


    # BST
    elif flags_obj.model == 'BST':
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=HIDDEN_SIZE,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = BSTModel(hparams, input_creator, seed=RANDOM_SEED)


    #DIEN
    elif flags_obj.model == 'DIEN':
        yaml_file = '../../reco_utils/recommender/deeprec/config/dien.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                hidden_size=HIDDEN_SIZE,
                                max_seq_length=max_seq_length,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                test_dropout=flags_obj.test_dropout,
                    )
        model = DIENModel(hparams, input_creator, seed=RANDOM_SEED)

    #Caser
    elif flags_obj.model == 'CASER':
        yaml_file = '../../reco_utils/recommender/deeprec/config/caser.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                T=1, n_v=128, n_h=128, L=3,
                                min_seq_length=1,
                                max_seq_length=max_seq_length,
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                test_dropout=flags_obj.test_dropout,
                    )
        model = CaserModel(hparams, input_creator)


    return model


class MailMaster(object):

    def __init__(self, flags_obj):

        username = flags_obj.mail_username
        receiver = flags_obj.receiver_address
        self.mail_host = flags_obj.mail_host
        self.port = flags_obj.mail_port
        self.mail_user = username
        self.mail_pass = self.get_mail_pass()
        self.sender = username
        self.receivers = [receiver]

        if 'kwai' in socket.gethostname():
            self.metrics = ['auc', 'wauc', 'mean_mrr', 'ndcg@1', 'ndcg@2', 'mean_alpha']
        else:
            self.metrics = ['auc', 'wauc', 'mean_mrr', 'ndcg@2', 'ndcg@4', 'ndcg@6', 'hit@2', 'hit@4', 'hit@6', 'mean_alpha']

    def get_mail_pass(self):

        connect_succeed = False
        max_connect_time = 10
        count = 0
        while not connect_succeed:
            count = count + 1
            if count > max_connect_time:
                raise ValueError("Wrong password for too many times.")
            mail_pass = getpass.getpass('Input mail password:')
            try:
                if 'kwai' in socket.gethostname():
                    smtpObj = smtplib.SMTP()
                else:
                    smtpObj = smtplib.SMTP_SSL(self.mail_host)
                smtpObj.connect(self.mail_host, self.port)
                login_status = smtpObj.login(self.mail_user, mail_pass) 
                if login_status[0] >= 200 and login_status[0] < 300:
                    connect_succeed = True
                    print('mail login success')
                    time.sleep(2)
                smtpObj.quit() 
            except smtplib.SMTPException as e:
                print('error',e)
        
        return mail_pass

    def get_table_head(self, res):

        table_head = ''.join(['<th>{}</th>'.format(metric) for metric in self.metrics if metric in res])

        return table_head

    def get_table_res(self, res):

        table_res = ''.join(['<th>{:.4f}</th>'.format(res[metric]) for metric in self.metrics if metric in res])

        return table_res

    def send_mail(self, flags_obj, mode, res):

        if mode in ['test', 'train']:
            table_head = self.get_table_head(res)
            table_res = self.get_table_res(res)
            html = """\
                <html>
                <body>
                    <p>
                        Mode: {}<br>
                        Exp name: {}<br>
                    </p>
                    <p>
                        <table border="1">
                            <tr>
                                {}
                            </tr>
                            <tr>
                                {}
                            </tr>
                        </table>
                    </p>
                </body>
                </html>
            """.format(mode, flags_obj.name, table_head, table_res)
        elif mode == 'counterfactual':
            if flags_obj.dataset in ['taobao', 'taobao_global']:
                table_head = self.get_table_head(res[0])
                table_res_strong_last = self.get_table_res(res[0])
                table_res_strong_first = self.get_table_res(res[1])
                table_res_weak = self.get_table_res(res[2])
                html = """\
                    <html>
                    <body>
                        <p>
                            Mode: {} {}<br>
                            Exp name: {}<br>
                        </p>
                        <p>
                            <br>Last Purchase<br>
                        </p>
                        <p>
                            <table border="1">
                                <tr>
                                    {}
                                </tr>
                                <tr>
                                    {}
                                </tr>
                            </table>
                        </p>
                        <p>
                            <br>First Purchase<br>
                        </p>
                        <p>
                            <table border="1">
                                <tr>
                                    {}
                                </tr>
                                <tr>
                                    {}
                                </tr>
                            </table>
                        </p>
                        <p>
                            <br>Last Click<br>
                        </p>
                        <p>
                            <table border="1">
                                <tr>
                                    {}
                                </tr>
                                <tr>
                                    {}
                                </tr>
                            </table>
                        </p>
                    </body>
                    </html>
                """.format(mode, flags_obj.test_counterfactual_mode, flags_obj.name, table_head, table_res_strong_last, table_head, table_res_strong_first, table_head, table_res_weak)
            if flags_obj.dataset == 'kuaishou_open':
                table_head = self.get_table_head(res[0])
                table_res_strong_like_last = self.get_table_res(res[0])
                table_res_strong_like_first = self.get_table_res(res[1])
                table_res_strong_follow_last = self.get_table_res(res[2])
                table_res_strong_follow_first = self.get_table_res(res[3])
                table_res_weak = self.get_table_res(res[4])
                html = """\
                    <html>
                    <body>
                        <p>
                            Mode: {} {}<br>
                            Exp name: {}<br>
                        </p>
                        <p>
                            <br>Last Like<br>
                        </p>
                        <p>
                            <table border="1">
                                <tr>
                                    {}
                                </tr>
                                <tr>
                                    {}
                                </tr>
                            </table>
                        </p>
                        <p>
                            <br>First Like<br>
                        </p>
                        <p>
                            <table border="1">
                                <tr>
                                    {}
                                </tr>
                                <tr>
                                    {}
                                </tr>
                            </table>
                        </p>
                        <p>
                            <br>Last Follow<br>
                        </p>
                        <p>
                            <table border="1">
                                <tr>
                                    {}
                                </tr>
                                <tr>
                                    {}
                                </tr>
                            </table>
                        </p>
                        <p>
                            <br>First Follow<br>
                        </p>
                        <p>
                            <table border="1">
                                <tr>
                                    {}
                                </tr>
                                <tr>
                                    {}
                                </tr>
                            </table>
                        </p>
                        <p>
                            <br>Last Click<br>
                        </p>
                        <p>
                            <table border="1">
                                <tr>
                                    {}
                                </tr>
                                <tr>
                                    {}
                                </tr>
                            </table>
                        </p>
                    </body>
                    </html>
                """.format(mode, flags_obj.test_counterfactual_mode, flags_obj.name, 
                            table_head, table_res_strong_like_last, table_head, table_res_strong_like_first,
                            table_head, table_res_strong_follow_last, table_head, table_res_strong_follow_first, table_head, table_res_weak)
        else:
            raise ValueError("not define this mode {0}".format(mode))

        message = MIMEText(html, "html", "utf-8")
        message['Subject'] = '{}: {}'.format(mode, flags_obj.name) 
        message['From'] = self.sender 
        message['To'] = ';'.join(self.receivers)

        try:
            if 'kwai' in socket.gethostname():
                smtpObj = smtplib.SMTP()
            else:
                smtpObj = smtplib.SMTP_SSL(self.mail_host)
            smtpObj.connect(self.mail_host, self.port)
            smtpObj.login(self.mail_user, self.mail_pass) 
            smtpObj.sendmail(self.sender, self.receivers, message.as_string()) 
            smtpObj.quit() 
            print('mail success')
        except smtplib.SMTPException as e:
            print('error',e)


def main(argv):

    flags_obj = FLAGS

    setproctitle.setproctitle('{}@LGY'.format(flags_obj.name))
    os.environ["CUDA_VISIBLE_DEVICES"] = flags_obj.gpu_id

    print("System version: {}".format(sys.version))
    print("Tensorflow version: {}".format(tf.__version__))

    if flags_obj.enable_mail_service:
        mail_master = MailMaster(flags_obj)

    print('start experiment')

    data_path_read = os.path.join(flags_obj.data_path, "data")
    if flags_obj.dataset == 'games':
        reviews_nameA = 'ratings_Video_Games.csv'
        meta_nameA = ''
    elif flags_obj.dataset == 'toys':
        reviews_nameA = 'ratings_Toys_and_Games.csv'
        meta_nameA = ''
    elif flags_obj.dataset == 'taobao':
        reviews_nameA = 'UserBehavior.csv'
        meta_nameA = ''
        strong_behavior_name = 'UserBuy.csv'
    elif flags_obj.dataset == 'yelp_global':
        reviews_nameA = 'yelp_academic_dataset_review.json'
        meta_nameA = 'yelp_academic_dataset_business.json'
    elif flags_obj.dataset == 'taobao_global':
        reviews_nameA = 'UserBehavior.csv'
        meta_nameA = ''
        strong_behavior_name = 'UserBuy.csv'
    elif flags_obj.dataset == 'kuaishou':
        reviews_nameA = 'filtered_main_single.csv'
        meta_nameA = ''
    elif flags_obj.dataset == 'kuaishou_fast':
        reviews_nameA = 'data_fast_single.csv'
        meta_nameA = ''

 
       # for test
    reviews_fileA = os.path.join(data_path_read, reviews_nameA)

    meta_fileA = os.path.join(data_path_read, meta_nameA)
    data_path = os.path.join(flags_obj.data_path, "negative_single_model/", flags_obj.dataset)

    train_fileA = os.path.join(data_path, r'train_dataA')
    valid_fileA = os.path.join(data_path, r'valid_dataA')
    test_fileA = os.path.join(data_path, r'test_dataA')
    test_file_neg = os.path.join(data_path, r'test_data_neg')
    train_file_pos = os.path.join(data_path, r'train_data_pos')

    user_vocab = os.path.join(data_path, r'user_vocab_cd.pkl')
    item_vocab = os.path.join(data_path, r'item_vocab_cd.pkl')
    cate_vocab = os.path.join(data_path, r'category_vocab_cd.pkl')
    output_fileA = os.path.join(data_path, r'outputA.txt')

    train_num_ngs = flags_obj.train_num_ngs
    valid_num_ngs = flags_obj.val_num_ngs
    test_num_ngs = flags_obj.test_num_ngs
    sample_rate = flags_obj.sample_rate

    input_files = [reviews_fileA, meta_fileA, train_fileA, valid_fileA, test_fileA, test_file_neg, train_file_pos, user_vocab, item_vocab, cate_vocab]

    if flags_obj.is_preprocess:
        data_preprocessing(*input_files, sample_rate=sample_rate, valid_num_ngs=valid_num_ngs, test_num_ngs=test_num_ngs, dataset=flags_obj.dataset)
    if not os.path.exists(test_fileA+'_group1'):
        if flags_obj.dataset == 'kuaishou':
            split_length = [50, 100, 200, 300]
        elif flags_obj.dataset == 'ks_cross_domain_fast':
            split_length = [50, 100, 150, 200]
        elif flags_obj.dataset == 'games':
            split_length = [10, 20, 30, 50]
        elif flags_obj.dataset == 'toys':
            split_length = [10, 15, 18, 20]
        elif flags_obj.dataset == 'taobao':
            split_length = [10, 20, 30, 50]

        group_sequence(test_file=test_fileA, split_length=split_length)

    save_path = os.path.join(flags_obj.save_path, flags_obj.model, flags_obj.dataset, flags_obj.name)
    if (flags_obj.is_onlyPos):
        model_pathA = os.path.join(save_path, "model_pos/", )
        summary_pathA = os.path.join(save_path, "summary_pos/")
        pretrain_pathA = os.path.join(save_path, "pretrain_pos/")
        train_fileA = train_file_pos


    else:
        model_pathA = os.path.join(save_path, "model/", )
        summary_pathA = os.path.join(save_path, "summary/")
        pretrain_pathA = os.path.join(save_path, "pretrain/") 

    finetune_path = flags_obj.finetune_path

    modelA = get_model(flags_obj, model_pathA, summary_pathA, pretrain_pathA, finetune_path, user_vocab, item_vocab, cate_vocab, train_num_ngs, data_path)


    if flags_obj.only_test:
        ckpt_pathA = tf.train.latest_checkpoint(model_pathA)
        modelA.load_model(ckpt_pathA)
        start_time = time.time()
        
        res_synA = modelA.run_weighted_eval(test_fileA, num_ngs=test_num_ngs)
        end_time = time.time()
        cost_time = end_time - start_time
        print('Time cost for testing is {0:.2f} mins'.format((cost_time)/60.0))
    
        print(flags_obj.name)
        print(res_synA)
        # print(res_syn_neg)
        for g in [1,2,3,4,5]:
            res_syn_group = modelA.run_weighted_eval(test_fileA+'_group'+str(g), num_ngs=test_num_ngs)
            print(flags_obj.name+'_group'+str(g))
            print(res_syn_group)

        if flags_obj.enable_mail_service:
            mail_master.send_mail(flags_obj, 'test', res)

        return
    if flags_obj.only_visualize:
        ckpt_pathA = tf.train.latest_checkpoint(model_pathA)
        modelA.load_model(ckpt_pathA)
        start_time = time.time()
        res_syn_train = modelA.run_weighted_eval_train(flags_obj.dataset, flags_obj.model, train_fileA, num_ngs=test_num_ngs)
        end_time = time.time()
        cost_time = end_time - start_time
        print('Time cost for testing is {0:.2f} mins'.format((cost_time)/60.0))
        return 


    if flags_obj.no_visual_host not in socket.gethostname():

        vm = None
    visual_path = os.path.join(save_path, "metrics/")
    tb = SummaryWriter(log_dir=visual_path, comment='tb')


    eval_metric = 'auc'
   
    start_time = time.time()
    print("I am Model A.")
    modelA = modelA.fit(train_fileA, valid_fileA, valid_num_ngs=valid_num_ngs, eval_metric=eval_metric, vm=vm, tb=tb, pretrain=flags_obj.pretrain) 
   
    end_time = time.time()
    cost_time = end_time - start_time
    print('Time cost for training is {0:.2f} mins'.format((cost_time)/60.0))

    ckpt_pathA = tf.train.latest_checkpoint(model_pathA)
    modelA.load_model(ckpt_pathA)
    

    start_time = time.time()
    res_synA = modelA.run_weighted_eval(test_fileA, num_ngs=test_num_ngs)

    end_time = time.time()
    cost_time = end_time - start_time
    print('Time cost for testing is {0:.2f} mins'.format((cost_time)/60.0))

    print(flags_obj.name)
    print(res_synA)


    if flags_obj.no_visual_host not in socket.gethostname():
        vm.show_test_info()
        vm.show_result(res_synA)

    tb.close()

    if flags_obj.enable_mail_service:
        mail_master.send_mail(flags_obj, 'train', res)

    if flags_obj.write_prediction_to_file:
        modelA = modelA.predict(test_file, output_file)


if __name__ == "__main__":
    
    app.run(main)
