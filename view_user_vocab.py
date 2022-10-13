from reco_utils.recommender.deeprec.deeprec_utils import cal_metric, cal_weighted_metric, cal_mean_alpha_metric, load_dict

user_vocab = '../../user_vocab.pkl'
dist = load_dict(user_vocab)
user_vocab_length = len(dist)
print(dist)
print(user_vocab_length)
