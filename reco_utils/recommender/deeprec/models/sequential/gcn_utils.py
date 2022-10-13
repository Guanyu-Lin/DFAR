# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import scipy.sparse as sp
import numpy as np

__all__ = ["GCN","GAT"]

def gcn_norm_edge(edge_index, num_nodes, edge_weight=None, renorm=True, improved=False, cache=None):
    cache_key = "gcn_normed_edge"

    if cache is not None and cache_key in cache and cache[cache_key] is not None:
        return cache[cache_key]

    if edge_weight is None:
        edge_weight = tf.ones([tf.shape(edge_index)[1]], dtype=tf.float32)

    fill_weight = 2.0 if improved else 1.0

    if renorm:
        edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes, edge_weight=edge_weight, fill_weight=fill_weight)

    row, col = edge_index[0], edge_index[1]
    deg = tf.math.unsorted_segment_sum(edge_weight, row, num_segments=num_nodes)
    deg_inv_sqrt = tf.pow(deg, -0.5)
    deg_inv_sqrt = tf.where(
        tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
        tf.zeros_like(deg_inv_sqrt),
        deg_inv_sqrt
    )

    normed_edge_weight = tf.gather(deg_inv_sqrt, row) * edge_weight * tf.gather(deg_inv_sqrt, col)

    if not renorm:
        edge_index, normed_edge_weight = add_self_loop_edge(edge_index, num_nodes, edge_weight=normed_edge_weight,
                                                            fill_weight=fill_weight)

    if cache is not None:
        cache[cache_key] = edge_index, normed_edge_weight

    return edge_index, normed_edge_weight


def gcn_mapper(repeated_x, neighbor_x, edge_weight=None):
    return neighbor_x * tf.expand_dims(edge_weight, 1)


def gcn(x, edge_index, edge_weight, kernel, bias=None, activation=None,
        renorm=True, improved=False, cache=None):
    """

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param edge_weight: Tensor or None, shape: [num_edges]
    :param kernel: Tensor, shape: [num_features, num_output_features], weight
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
    :param improved: Whether use improved GCN or not.
    :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
    :return: Updated node features (x), shape: [num_nodes, num_output_features
    """

    num_nodes = tf.shape(x)[0]
    updated_edge_index, normed_edge_weight = gcn_norm_edge(edge_index, num_nodes, edge_weight, renorm, improved, cache)

    x = x @ kernel

    h = aggregate_neighbors(
        x, updated_edge_index, normed_edge_weight,
        gcn_mapper,
        sum_reducer,
        identity_updater
    )

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h


def add_self_loop_edge(edge_index, num_nodes, edge_weight=None, fill_weight=1.0):

    diagnal_edge_index = tf.stack([tf.range(num_nodes, dtype=tf.int32)] * 2, axis=0)
    updated_edge_index = tf.concat([edge_index, diagnal_edge_index], axis=1)

    if not tf.is_tensor(edge_index):
        updated_edge_index = updated_edge_index.numpy()

    if edge_weight is not None:
        diagnal_edge_weight = tf.cast(tf.fill([num_nodes], fill_weight), tf.float32)
        updated_edge_weight = tf.concat([edge_weight, diagnal_edge_weight], axis=0)

        if not tf.is_tensor(edge_weight):
            updated_edge_weight = updated_edge_weight.numpy()
    else:
        updated_edge_weight = None

    return updated_edge_index, updated_edge_weight


#  def aggregate_neighbors(x, edge_index, edge_weight=None, mapper=identity_mapper,
                        #  reducer=sum_reducer, updater=sum_updater):
def aggregate_neighbors(x, edge_index, edge_weight, mapper,
                        reducer, updater):
    """
    :param x:
    :param edge_index:
    :param mapper: (features_of_node, features_of_neighbor_node, edge_weight) => neighbor_msg
    :param reducer: (neighbor_msg, node_index) => reduced_neighbor_msg
    :param updater: (features_of_node, reduced_neighbor_msg, num_nodes) => aggregated_node_features
    :return:
    """

    # 边索引为空，则返回原特征矩阵
    if tf.shape(edge_index)[0] == 0:
        return x

    row, col = edge_index[0], edge_index[1]

    repeated_x = tf.gather(x, row)
    neighbor_x = tf.gather(x, col)

    neighbor_msg = mapper(repeated_x, neighbor_x, edge_weight=edge_weight)
    reduced_msg = reducer(neighbor_msg, row, num_nodes=tf.shape(x)[0])
    udpated = updater(x, reduced_msg)

    return udpated

def sum_updater(x, reduced_neighbor_msg):
    return x + reduced_neighbor_msg

def sum_reducer(neighbor_msg, node_index, num_nodes=None):
    return tf.math.unsorted_segment_sum(neighbor_msg, node_index, num_segments=num_nodes)

def identity_updater(x, reduced_neighbor_msg):
    return reduced_neighbor_msg

def identity_updater(x, reduced_neighbor_msg):
    return reduced_neighbor_msg

class GCN:
    """
    Graph Convolutional Layer
    """

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def __init__(self, units, activation=None, use_bias=True,
                 renorm=True, improved=False,
                 kernel_regularizer=None, bias_regularizer=None, *args, **kwargs):
        """

        :param units: Positive integer, dimensionality of the output space.
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
        :param improved: Whether use improved GCN or not.
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        """
        super().__init__(*args, **kwargs)
        self.units = units

        self.acvitation = activation
        self.use_bias = use_bias

        self.kernel = None
        self.bias = None

        self.renorm = renorm
        self.improved = improved

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        return gcn(x, edge_index, edge_weight, self.kernel, self.bias,
                   activation=self.acvitation, renorm=self.renorm, improved=self.improved, cache=cache)


class GAT:

    def __init__(self, units,
                 attention_units=None,
                 activation=None,
                 use_bias=True,
                 num_heads=1,
                 query_activation=tf.nn.relu,
                 key_activation=tf.nn.relu,
                 drop_rate=0.0,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 *args, **kwargs):
        """

        :param units: Positive integer, dimensionality of the output space.
        :param attention_units: Positive integer, dimensionality of the output space for Q and K in attention.
        :param activation: Activation function to use.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param num_heads: Number of attention heads.
        :param query_activation: Activation function for Q in attention.
        :param key_activation: Activation function for K in attention.
        :param drop_rate: Dropout rate.
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        :param bias_regularizer: Regularizer function applied to the bias vector.
        """
        super().__init__(*args, **kwargs)
        self.units = units
        self.attention_units = units if attention_units is None else attention_units
        self.drop_rate = drop_rate

        self.query_kernel = None
        self.query_bias = None
        self.query_activation = query_activation

        self.key_kernel = None
        self.key_bias = None
        self.key_activation = key_activation

        self.kernel = None
        self.bias = None

        self.acvitation = activation
        self.use_bias = use_bias
        self.num_heads = num_heads

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.query_kernel = self.add_weight("query_kernel", shape=[num_features, self.attention_units],
                                            initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.query_bias = self.add_weight("query_bias", shape=[self.attention_units],
                                          initializer="zeros", regularizer=self.bias_regularizer)

        self.key_kernel = self.add_weight("key_kernel", shape=[num_features, self.attention_units],
                                          initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.key_bias = self.add_weight("key_bias", shape=[self.attention_units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

        self.kernel = self.add_weight("kernel", shape=[num_features, self.units],
                                      initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index]
        :return: Updated node features (x), shape: [num_nodes, units]
        """
        x, edge_index = inputs

        return gat(x, edge_index,
                   self.query_kernel, self.query_bias, self.query_activation,
                   self.key_kernel, self.key_bias, self.key_activation,
                   self.kernel, self.bias, self.acvitation,
                   num_heads=self.num_heads,
                   drop_rate=self.drop_rate,
                   training=training)

# follow Transformer-Style Attention
# Attention is all you need
def gat(x, edge_index,
        query_kernel, query_bias, query_activation,
        key_kernel, key_bias, key_activation,
        kernel, bias=None, activation=None, num_heads=1, drop_rate=0.0, training=False):
    """

    :param x: Tensor, shape: [num_nodes, num_features], node features
    :param edge_index: Tensor, shape: [2, num_edges], edge information
    :param query_kernel: Tensor, shape: [num_features, num_query_features], weight for Q in attention
    :param query_bias: Tensor, shape: [num_query_features], bias for Q in attention
    :param query_activation: Activation function for Q in attention.
    :param key_kernel: Tensor, shape: [num_features, num_key_features], weight for K in attention
    :param key_bias: Tensor, shape: [num_key_features], bias for K in attention
    :param key_activation: Activation function for K in attention.
    :param kernel: Tensor, shape: [num_features, num_output_features], weight
    :param bias: Tensor, shape: [num_output_features], bias
    :param activation: Activation function to use.
    :param num_heads: Number of attention heads.
    :param drop_rate: Dropout rate.
    :param training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    :return: Updated node features (x), shape: [num_nodes, num_output_features]
    """


    num_nodes = tf.shape(x)[0]

    # self-attention
    edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]

    Q = query_activation(x @ query_kernel + query_bias)
    Q = tf.gather(Q, row)

    K = key_activation(x @ key_kernel + key_bias)
    K = tf.gather(K, col)

    V = x @ kernel

    # xxxxx_ denotes the multi-head style stuff
    # 在最后一维切分，在第0维拼接。为了不增加参数量
    Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)
    # 利用节点数更新边索引，在第一维拼接，不是最后一维
    edge_index_ = tf.concat([edge_index + i * num_nodes for i in range(num_heads)], axis=1)

    att_score_ = tf.reduce_sum(Q_ * K_, axis=-1)
    normed_att_score_ = segment_softmax(att_score_, edge_index_[0], num_nodes * num_heads)

    if training and drop_rate > 0.0:
        normed_att_score_ = tf.compat.v2.nn.dropout(normed_att_score_, drop_rate)

    h_ = aggregate_neighbors(
        V_, edge_index_, normed_att_score_,
        gcn_mapper,
        sum_reducer,
        identity_updater
    )

    # 将多头再纵向拆分，横向拼接回来
    h = tf.concat(tf.split(h_, num_heads, axis=0), axis=-1)

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h


def segment_softmax(data, segment_ids, num_segments):
    max_values = tf.math.unsorted_segment_max(data, segment_ids, num_segments=num_segments)
    gathered_max_values = tf.gather(max_values, segment_ids)
    exp = tf.exp(data - tf.stop_gradient(gathered_max_values))
    denominator = tf.math.unsorted_segment_sum(exp, segment_ids, num_segments=num_segments) + 1e-8
    gathered_denominator = tf.gather(denominator, segment_ids)
    score = exp / gathered_denominator
    return score


def init_gat_weight(hidden_dim, n_layers):
    all_weights = dict()
    initializer = tf.random_normal_initializer(stddev=0.01) #tf.contrib.layers.xavier_initializer()
    weight_size_list = [hidden_dim]*(n_layers +1) 
    for k in range(n_layers):
        all_weights['query_kernel_%d' %k] = tf.Variable(
            initializer([weight_size_list[k], weight_size_list[k+1]]), name='query_kernel_%d' % k)
        all_weights['query_bias_%d' %k] = tf.Variable(
            initializer([weight_size_list[k+1]]), name='query_bias_%d' % k)
        all_weights['key_kernel_%d' %k] = tf.Variable(
            initializer([weight_size_list[k], weight_size_list[k+1]]), name='key_kernel_%d' % k)
        all_weights['key_bias_%d' %k] = tf.Variable(
            initializer([weight_size_list[k+1]]), name='key_bias_%d' % k)
        all_weights['kernel_%d' %k] = tf.Variable(
            initializer([weight_size_list[k], weight_size_list[k+1]]), name='kernel_%d' % k)
        all_weights['bias_%d' %k] = tf.Variable(
            initializer([weight_size_list[k+1]]), name='bias_%d' % k)
    return all_weights 

def init_gcn_weight(hidden_dim, n_layers):

    all_weights = dict()
    initializer = tf.random_normal_initializer(stddev=0.01) #tf.contrib.layers.xavier_initializer()
    weight_size_list = [hidden_dim]*(n_layers +1) 
#          if self.pretrain_data is None:
    #      all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
    #      all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
    #      print('using random initialization')#print('using xavier initialization')
    #  else:
    #      all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
    #                                                  name='user_embedding', dtype=tf.float32)
    #      all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
    #                                                  name='item_embedding', dtype=tf.float32)
#              print('using pretrained initialization')
    for k in range(n_layers):
        all_weights['W_gc_%d' %k] = tf.Variable(
            initializer([weight_size_list[k], weight_size_list[k+1]]), name='W_gc_%d' % k)
        all_weights['b_gc_%d' %k] = tf.Variable(
            initializer([weight_size_list[k+1]]), name='b_gc_%d' % k)
    return all_weights

def init_score_gcn_weight(hidden_dim, n_layers):

    all_weights = dict()
    initializer = tf.random_normal_initializer(stddev=0.01) #tf.contrib.layers.xavier_initializer()
    weight_size_list = [hidden_dim]*n_layers 
#          if self.pretrain_data is None:
    #      all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
    #      all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
    #      print('using random initialization')#print('using xavier initialization')
    #  else:
    #      all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
    #                                                  name='user_embedding', dtype=tf.float32)
    #      all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
    #                                                  name='item_embedding', dtype=tf.float32)
#              print('using pretrained initialization')
    for k in range(n_layers):
        all_weights['W_sc_%d' %k] = tf.Variable(
            initializer([weight_size_list[k], 1]), name='W_sc_%d' % k)
        all_weights['b_sc_%d' %k] = tf.Variable(
            initializer([1]), name='b_sc_%d' % k)
    return all_weights

def topk_pool(source_index, score, k=None, ratio=None):
    """
    功能：source_index是节点所属graph/batch，score是节点分数的分数，将每个graph/batch中score最高的k个的节点返回

    :param sorted_source_index: index of source node (of edge) or source graph (of node)
    源节点（边缘）或源图形（节点）的索引
    :param sorted_score: 1-D Array
     一维数组
    :param k:
    :param ratio:
    :return: sampled_edge_index, sampled_edge_score, sample_index
    """

    # k和ration只能也必须输入一个
    if k is None and ratio is None:
        raise Exception("you should provide either k or ratio for topk_pool")
    elif k is not None and ratio is not None:
        raise Exception("you should provide either k or ratio for topk_pool, not both of them")

    # currently, we consider the source_index is not sorted
    # the option is preserved for future performance optimization
    #  当前，我们认为source_index未排序
    #  该选项被保留以用于将来的性能
    source_index_sorted = False

    if source_index_sorted:
        sorted_source_index = source_index
        # sort score by source_index
        sorted_score = score
    else:
        # 将graph/batch索引排序，返回排序好的graph/batch索引index和对应的节点分数score
        source_index_perm = tf.argsort(source_index)
        sorted_source_index = tf.gather(source_index, source_index_perm)
        sorted_score = tf.gather(score, source_index_perm)


    # 节点分数score展开成1维（原来可能是2维，但多出的一维为空；若原来为一维，则不进行操作）
    sorted_score = tf.reshape(sorted_score, [-1])

    # 节点的数量
    num_targets = union_len(sorted_source_index)
    target_ones = tf.ones([num_targets], dtype=tf.int32)
    # 每个graph的节点数
    num_targets_for_sources = tf.math.segment_sum(target_ones, sorted_source_index)
    # number of columns for score matrix
    # 具有最多节点的图的节点数，设为分数矩阵的列数
    num_cols = tf.reduce_max(num_targets_for_sources)

    # max index of source + 1
    # graph/batch的数量
    num_seen_sources = tf.shape(num_targets_for_sources)[0]

    # 最小分数，用于矩阵初始化
    min_score = tf.reduce_min(sorted_score)

    # 节点数的累加，去掉最后一个元素，前面加一个0
    num_targets_before = tf.concat([
        tf.zeros([1], dtype=tf.int32),
        tf.math.cumsum(num_targets_for_sources)[:-1]
    ], axis=0)

    # 每个节点在排序后的序列中，以第一个相同节点为起点的索引
    target_index_for_source = tf.range(0, num_targets) - tf.gather(num_targets_before, sorted_source_index)

    # 分数矩阵，行为节点数，列为节点出现最大数，元素填充为最小分数-1（就是填充比最小的数还小的元素，因为不确定最小的数是多少）
    score_matrix = tf.cast(tf.fill([num_seen_sources, num_cols], min_score - 1.0), dtype=tf.float32)
    # 分数索引，行索引为节点数，列索引为节点重复出现的次数
    score_index = tf.stack([sorted_source_index, target_index_for_source], axis=1)
    # 更新分数矩阵，根据分数索引和分数向量
    score_matrix = tf.tensor_scatter_nd_update(score_matrix, score_index, sorted_score)

    # 按列倒序，返回矩阵，行为每个graph，列为分数最高的原分数矩阵的列索引
    sort_index = tf.argsort(score_matrix, axis=-1, direction="DESCENDING")

    if k is not None:
        # graph数量长度的向量，若k小于该graph的节点数量，则元素为k，否则为节点数量
        node_k = tf.math.minimum(
            tf.cast(tf.fill([num_seen_sources], k), dtype=tf.int32),
            num_targets_for_sources
        )
    else:
        # graph数量长度的向量，每个元素为该graph节点数量乘以ratio比值（最小为1，因为ceil取了上界），每个graph至少保留一个节点
        node_k = tf.cast(
            tf.math.ceil(tf.cast(num_targets_for_sources, dtype=tf.float32) * tf.cast(ratio, dtype=tf.float32)),
            dtype=tf.int32
        )

    # 定义动态tensor array
    left_k_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True, element_shape=[2])
    num_rows = tf.shape(node_k)[0]

    #  # 每行要取的列数不同，所以只能循环。先循环每行，然后循环要取的列，在array中填写行列索引。current_size控制array的长度
    #  current_size = 0
    #  for row_index in range(num_rows):
    #      num_cols = node_k[row_index]
    #      for col_index in range(num_cols):
    #          left_k_index = left_k_index.write(current_size, [row_index, col_index])
            #  current_size += 1

    ## for循环改写为while_loop
    def cond1(current_size, left_k_index, row_index, col_index):
        return row_index < num_rows
    def body1(current_size, left_k_index, row_index, col_index):
        def cond2(current_size, left_k_index, row_index, col_index):
            num_cols = node_k[row_index]
            return col_index < num_cols
        def body2(current_size, left_k_index, row_index, col_index):
            left_k_index = left_k_index.write(current_size, [row_index, col_index])
            current_size += tf.ones([], tf.int32)
            col_index += tf.ones([], tf.int32)
            return current_size, left_k_index, row_index, col_index 
        current_size, left_k_index, row_index, col_index = tf.while_loop(cond2, body2, [current_size, left_k_index, row_index, col_index])
        row_index += tf.ones([], tf.int32)
        return  current_size, left_k_index, row_index, col_index 
    current_size, row_index, col_index = tf.zeros([], tf.int32), tf.zeros([], tf.int32), tf.zeros([], tf.int32)
    current_size, left_k_index, row_index, col_index = tf.while_loop(cond1, body1, [current_size, left_k_index, row_index, col_index])

    #  将TensorArray中元素叠起来当做一个Tensor输出
    #  https://blog.csdn.net/z2539329562/article/details/80639199
    import pdb; pdb.set_trace()
    left_k_index = left_k_index.stack()

    # left_k_index = [[row_index, col_index]
    #                 for row_index, num_cols in enumerate(node_k)
    #                 for col_index in range(num_cols)]

    # left_k_index = tf.convert_to_tensor(left_k_index, dtype=tf.int32)

    # 目标节点在分数矩阵中的行列索引
    # 行索引一致，然后从分数矩阵中拿列索引
    sample_col_index = tf.gather_nd(sort_index, left_k_index)
    sample_row_index = left_k_index[:, 0]
    #  sample_row_index = left_k_index[0]

    # 目标节点在排序后的score向量中的索引，根据graph索引，确定起点，然后加上重复次数，即增量。
    topk_index = tf.gather(num_targets_before, sample_row_index) + sample_col_index

    if source_index_sorted:
        return topk_index
    else:
        # 返回排序前的score向量索引
        #  return tf.gather(source_index_perm, topk_index)
        #  return tf.gather(source_index_perm, topk_index), sample_row_index
        #  return tf.gather(source_index_perm, topk_index), node_k
        return tf.gather(source_index_perm, topk_index), node_k, current_size, row_index

    return topk_index

def union_len(data):
    if tf.is_tensor(data):
        #  return data.shape.as_list()[0]
        return tf.shape(data)[0]
    else:
        return len(data) 

def dense2sparse(self, dense):
    idx = tf.where(tf.not_equal(dense, 0))
    sparse = tf.SparseTensor(idx, tf.gather_nd(dense, idx), dense.get_shape())
    return sparse


def sample_new_graph_by_node_index(x, edge_index, edge_weight, node_graph_index, sampled_node_index):
    """

    :param sampled_node_index: Tensor/NDArray, shape: [num_sampled_nodes]
    :return: A new cloned graph where nodes that are not in sampled_node_index are removed,
        as well as the associated information, such as edges.
    """
    #  is_batch_graph = isinstance(self, BatchGraph)

    #  x = self.x
    #  edge_index = self.edge_index
    #  y = self.y
    #  edge_weight = self.edge_weight
    # 如果是batch_graph类，则还要多考虑两个额外参数
    #  if is_batch_graph:
        #  node_graph_index = self.node_graph_index
        #  edge_graph_index = self.edge_graph_index

    # 根据sampled_node_index，索引tensor
#      def sample_common_data(data):
        #  if data is not None:
        #      data_is_tensor = tf.is_tensor(data)
        #      if data_is_tensor:
        #          data = tf.gather(data, sampled_node_index)
        #      else:
        #          data = convert_union_to_numpy(data)
        #          data = data[sampled_node_index]
        #
        #      # 为什么是tensor，还要再转换为tensor？
        #      if data_is_tensor:
        #          data = tf.convert_to_tensor(data)
#          return data

    # 根据sampled_node_index，索引节点特征x，graph标签y，节点图标签node_graph_index
    #  x = sample_common_data(x)
    x = tf.gather(x, sampled_node_index)
    #  y = sample_common_data(y)
    #  if is_batch_graph:
    #  node_graph_index = sample_common_data(node_graph_index)
    node_graph_index = tf.gather(node_graph_index, sampled_node_index)

    if edge_index is not None:

        #  将节点索引转换为numpy（这numpy的转换貌似在tf1.14中无法实现，而且其实tensor也可以进行边索引，参照我之前写的intersection）
        #  sampled_node_index = convert_union_to_numpy(sampled_node_index)

        # 将不存在在node_index中的的edge_index mask掉
        #  edge_index_is_tensor = tf.is_tensor(edge_index)
        #  edge_index = convert_union_to_numpy(edge_index)
        edge_mask = compute_edge_mask_by_node_index(edge_index, sampled_node_index)

        # 只含有采样节点的边索引，和对应的行、列
        #  edge_index = edge_index[:, edge_mask]
        edge_index = tf.gather(edge_index, tf.squeeze(tf.where(edge_mask), -1))
        #  row, col = edge_index
        #  row, col = edge_index[:,0], edge_index[:,1]
        row, col = edge_index[0], edge_index[1]

        # 将边索引按采样后的节点从小到大重新排列（因为graph的节点和节点特征都缩减了，后面还要用边索引这些特征）
        #  max_sampled_node_index = np.max(sampled_node_index) + 1
        max_sampled_node_index = tf.reduce_max(sampled_node_index) + 1
        #  new_node_range = list(range(len(sampled_node_index)))
        new_node_range = tf.range(0, tf.shape(sampled_node_index)[0])
        # 节点数量长的一维向量，其余所有元素初始化为-1，将采样节点对应位置填充未排序后的元素
        #  reverse_index = np.full([max_sampled_node_index + 1], -1, dtype=np.int32)
        # 这里加一不加一都可？
        reverse_index = tf.fill([max_sampled_node_index + 1], -1) 
        #  reverse_index[sampled_node_index] = new_node_range
        reverse_index = tf.tensor_scatter_nd_update(reverse_index, sampled_node_index, new_node_range)

        # 获取重排后的行列
        #  row = reverse_index[row]
        #  col = reverse_index[col]
        row = tf.gather(reverse_index, row)
        col = tf.gather(reverse_index, col)
        #  edge_index = np.stack([row, col], axis=0)
        #  edge_index = tf.stack([row, col], axis=1)
        edge_index = tf.stack([row, col], axis=0)
        #  if edge_index_is_tensor:
            #  edge_index = tf.convert_to_tensor(edge_index)

        #  def sample_by_edge_mask(data):
            #  if data is not None:
                #  data_is_tensor = tf.is_tensor(data)
                #  data = convert_union_to_numpy(data)
                #  data = data[edge_mask]
                #  if data_is_tensor:
                    #  data = tf.convert_to_tensor(data)
            #  return data

        # 将被mask的边索引的边权重也丢掉
        #  edge_weight = sample_by_edge_mask(edge_weight)
        edge_weight = tf.gather(edge_weight, tf.where(edge_mask))
        # 将被mask的边索引的边-图对应关系也丢掉
        #  if is_batch_graph:
        #  edge_graph_index = sample_by_edge_mask(edge_graph_index)

    #  if is_batch_graph:
        #  return BatchGraph(x=x, edge_index=edge_index, node_graph_index=node_graph_index,
                          #  edge_graph_index=edge_graph_index, y=y, edge_weight=edge_weight)
    #  else:
        #  return Graph(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)

    return x, edge_index, edge_weight, node_graph_index 

#  def convert_union_to_numpy(data, dtype=None):
    #  if data is None:
        #  return data
#
    #  if tf.is_tensor(data):
        #  np_data = data.numpy()
    #  elif isinstance(data, list):
        #  np_data = np.array(data)
    #  else:
        #  np_data = data
#
    #  if dtype is not None:
        #  np_data = np_data.astype(dtype)
#
    #  return np_data


def compute_edge_mask_by_node_index(edge_index, node_index):
    """
    输入原始边索引(2*E)，新的节点索引(N)，返回边索引的mask(E)（节点在边中为true，否则为false）
    """

    #  edge_index_is_tensor = tf.is_tensor(edge_index)

    #  node_index = convert_union_to_numpy(node_index)
    #  edge_index = convert_union_to_numpy(edge_index)

    #  max_node_index = np.maximum(np.max(edge_index), np.max(node_index))
    max_node_index = tf.maximum(tf.reduce_max(edge_index), tf.reduce_max(node_index))
    #  node_mask = np.zeros([max_node_index + 1]).astype(np.bool)
    node_mask = tf.zeros([max_node_index + 1], tf.bool)
    #  node_mask[node_index] = True
    node_mask = tf.tensor_scatter_nd_update(node_mask, node_index, tf.ones([1], tf.bool))
    #  row, col = edge_index
    #  row, col = edge_index[:,0], edge_index[:,1]
    row, col = edge_index[0], edge_index[1]
    #  row_mask = node_mask[row]
    #  col_mask = node_mask[col]
    row_mask = tf.gather(node_mask, row)
    col_mask = tf.gather(node_mask, col)
    # 逻辑与，一个节点不存在，则将整条边mask
    #  edge_mask = np.logical_and(row_mask, col_mask)
    edge_mask = tf.logical_and(row_mask, col_mask)

    # 如果输入是tensor，则返回tensor
    #  if edge_index_is_tensor:
        #  edge_mask = tf.convert_to_tensor(edge_mask, dtype=tf.bool)

    return edge_mask


def mean_pool(x, node_graph_index, num_graphs=None):
    '''
    每个节点属于哪个图，进行平均池化
    x：节点特征矩阵
    node_graph_index：一维向量，key为节点id，value为图id
    '''
    # 如果没有参数‘图的数量，则图的数量为最大索引+1’
    if num_graphs is None:
        num_graphs = tf.reduce_max(node_graph_index) + 1
    # 每个graph的节点数，一维，key为graph id，value为graph的node数量
    num_nodes_of_graphs = segment_count(node_graph_index, num_segments=num_graphs)
    # 每个graph的节点特征求和
    sum_x = tf.math.unsorted_segment_sum(x, node_graph_index, num_segments=num_graphs)
    # 求平均
    return sum_x / (tf.cast(tf.expand_dims(num_nodes_of_graphs, -1), tf.float32) + 1e-8)


def sum_pool(x, node_graph_index, num_graphs=None):
    if num_graphs is None:
        num_graphs = tf.reduce_max(node_graph_index) + 1
    sum_x = tf.math.unsorted_segment_sum(x, node_graph_index, num_segments=num_graphs)
    return sum_x


def max_pool(x, node_graph_index, num_graphs=None):
    if num_graphs is None:
        num_graphs = tf.reduce_max(node_graph_index) + 1
    # max_x = tf.math.unsorted_segment_max(x, node_graph_index, num_segments=num_graphs)
    max_x = segment_op_with_pad(tf.math.segment_max, x, node_graph_index, num_segments=num_graphs)
    return max_x


def min_pool(x, node_graph_index, num_graphs=None):
    if num_graphs is None:
        num_graphs = tf.reduce_max(node_graph_index) + 1
    # min_x = tf.math.unsorted_segment_min(x, node_graph_index, num_segments=num_graphs)
    min_x = segment_op_with_pad(tf.math.segment_min, x, node_graph_index, num_segments=num_graphs)
    return min_x


def segment_count(index, num_segments=None):
    '''
    分段计数
    index：输入一维索引向量
    '''
    data = tf.ones_like(index)
    if num_segments is None:
        #  如果没有要求片段数量，则沿张量的片段计算总和。
        return tf.math.segment_sum(data, index)
    else:
        # 如果给定段 ID i 的和为空,则 output[i] = 0
        #  https://www.cnblogs.com/gaofighting/p/9706081.html
        return tf.math.unsorted_segment_sum(data, index, num_segments=num_segments)

def segment_op_with_pad(segment_op, data, segment_ids, num_segments):
    reduced_data = segment_op(data, segment_ids)
    num_paddings = num_segments - tf.shape(reduced_data)[0]

    pads = tf.zeros([num_paddings] + data.shape.as_list()[1:], dtype=reduced_data.dtype)
    outputs = tf.concat(
        [reduced_data, pads],
        axis=0
    )
    return outputs


#  def segment_top_k(x, I, ratio, top_k_var):
def segment_top_k(x, I, ratio):
    """
    Returns indices to get the top K values in x segment-wise, according to
    返回索引，以获取x分段前K个值，根据
    the segments defined in I. K is not fixed, but it is defined as a ratio of
     I中定义的分段。K不是固定的，但定义为
    the number of elements in each segment.
     每个段中的元素数的ratio。
    :param x: a rank 1 Tensor;
     1级张量;
    :param I: a rank 1 Tensor with segment IDs for x;
     1级张量，是x的段ID;
    :param ratio: float, ratio of elements to keep for each segment;
     float，每个段要保留的元素比例；
    :param top_k_var: a tf.Variable created without shape validation (i.e.,
     一个不带形状验证的tf变量（即，
    `tf.Variable(0.0, validate_shape=False)`);
    :return: a rank 1 Tensor containing the indices to get the top K values of
     返回：1级张量，包含能获取的x中的每段前K个值的索引。
    each segment in x.
    """
    ## 计算各种参数
    I = tf.cast(I, tf.int32)
    num_nodes = tf.math.segment_sum(tf.ones_like(I), I)  # Number of nodes in each graph
    cumsum = tf.cumsum(num_nodes)  # Cumulative number of nodes (A, A+B, A+B+C)
    cumsum_start = cumsum - num_nodes  # Start index of each graph
    n_graphs = tf.shape(num_nodes)[0]  # Number of graphs in batch
    max_n_nodes = tf.reduce_max(num_nodes)  # Order of biggest graph in batch
    batch_n_nodes = tf.shape(I)[0]  # Number of overall nodes in batch
    to_keep = tf.math.ceil(ratio * tf.cast(num_nodes, tf.float32))
    to_keep = tf.cast(to_keep, I.dtype)  # Nodes to keep in each graph

    index = tf.range(batch_n_nodes)
    #  tf.gather(cumsum_start, I)表示每个node对应的graph的累计开始索引
    #  (index - tf.gather(cumsum_start, I))表示每个node对应开始索引的增量
    #  (I * max_n_nodes)表示每个graph的节点编号向后都移动最大节点数
    index = (index - tf.gather(cumsum_start, I)) + (I * max_n_nodes)

    y_min = tf.reduce_min(x)
    # 一阶，维度为图的数量*图中最大节点数量
    dense_y = tf.ones((n_graphs * max_n_nodes,))
    # subtract 1 to ensure that filler values do not get picked
    #  元素填充为最小分数-1（就是填充比最小的数还小的元素，因为不确定最小的数是多少）
    dense_y = dense_y * tf.cast(y_min - 1, dense_y.dtype)
    #  dense_y = tf.cast(dense_y, top_k_var.dtype)
    # top_k_var is a variable with unknown shape defined in the elsewhere
    # 赋值操作，将常量赋值给变量
    #  top_k_var.assign(dense_y)
    #  index[..., None] 通过切片添加轴
    #  dense_y = tf.tensor_scatter_nd_update(top_k_var, index[..., None], tf.cast(x, top_k_var.dtype)) # 没有必要引入变量吧
    #  import pdb; pdb.set_trace()
    dense_y = tf.tensor_scatter_nd_update(dense_y, index[..., None], tf.cast(x, dense_y.dtype)) # index至少要有两个轴
    # 将一阶张量转换为二阶
    dense_y = tf.reshape(dense_y, (n_graphs, max_n_nodes))

    # 按列倒序，返回矩阵，行为每个graph，列为分数最高的原分数矩阵的列索引
    perm = tf.argsort(dense_y, direction='DESCENDING')
    # 因为返回的是矩阵中的列索引，所以还要加上每个graph的开始索引，以便后面展平
    perm = perm + cumsum_start[:, None]
    perm = tf.reshape(perm, (-1,))

    ## 生成topk节点数量的mask，每个graph的topk不同
    # 将[1,0]在第一维度上复制n_graph次，即dim = (2*n_graph)
    to_rep = tf.tile(tf.constant([1., 0.]), (n_graphs,))
    # 每个graph要保留的节点数和要删除的节点数，先concat，然后把所有graph展平，即dim = (2*n_graph)
    rep_times = tf.reshape(tf.concat((to_keep[:, None], (max_n_nodes - to_keep)[:, None]), -1), (-1,))
    # 把1复制“要保留的节点数”次，把0复制“要删除的节点数”次
    mask = repeat(to_rep, rep_times)

    # 只留下mask元素的索引
    perm = tf.boolean_mask(perm, mask)

    return perm

def repeat(x, repeats):
    """
    Repeats elements of a Tensor (equivalent to np.repeat, but only for 1D
    重复张量的元素（等效于np.repeat，但仅适用于一维张量）。
    tensors).
    :param x: rank 1 Tensor;
     ：1级张量;
    :param repeats: rank 1 Tensor with same shape as x, the number of
     ：1级张量，其形状与x相同，每个元素的重复的数量；
    repetitions for each element;
    :return: rank 1 Tensor, of shape `(sum(repeats), )`.
     ：返回：1阶张量，形状为（sum（repeats）,）。
    """
    x = tf.expand_dims(x, 1)
    # 最大重复数，用于后面range
    max_repeats = tf.reduce_max(repeats)
    tile_repeats = [1, max_repeats]
    # 将目标tensor在第二维度上重复最大重复次数
    arr_tiled = tf.tile(x, tile_repeats)
    # 小于返回真，大于返回假
    # 在目标repeats前的mask为1，在目标repeats到最大repeats之间的mask为0
    mask = tf.less(tf.range(max_repeats), tf.expand_dims(repeats, 1))
    # 提取mask，并展平
    result = tf.reshape(tf.boolean_mask(arr_tiled, mask), [-1])
    return result


def sparse_slice(indices, values, needed_row_ids):
  needed_row_ids = tf.reshape(needed_row_ids, [1, -1])
  num_rows = tf.shape(indices)[0]
  partitions = tf.cast(tf.reduce_any(tf.equal(tf.reshape(indices[:,0], [-1, 1]), needed_row_ids), 1), tf.int32)
  rows_to_gather = tf.dynamic_partition(tf.range(num_rows), partitions, 2)[1]
  slice_indices = tf.gather(indices, rows_to_gather)
  slice_values = tf.gather(values, rows_to_gather)
  return slice_indices, slice_values

def maximum_aggregator1(A, X):
    # 显存爆炸
    # X: B*L*F; A: B*L*L
    output_shape = X.get_shape()
    graph_num = tf.shape(X)[0] # B
    node_num = tf.shape(X)[1] # L

    flat_A = tf.reshape(A,[graph_num,-1,1]) # B*(L*L)*1
    tiled_X = tf.tile(X,[1, node_num, 1]) # B*(L*L)*F
    flat_X_dot_A = tf.reshape(tiled_X*flat_A - 1e4*(1-flat_A),[graph_num,node_num,node_num,-1]) # B*L*L*F
    output_X = tf.reduce_max(flat_X_dot_A,axis=2,keepdims=False) # B*L*F

    # 作用：生成计算图时，后面的conv1d等函数需要维度确定
    output_X.set_shape(output_shape)
    return output_X

def maximum_aggregator(A, X):
    # X: B*L*F; A: B*L*L
    output_shape = X.get_shape()
    graph_num = tf.shape(X)[0] # B
    node_num = tf.shape(X)[0]
    output_dim = int(output_shape[-1])

    output_X = tf.zeros([0,output_dim])
    _,_,_,output_X = tf.while_loop(lambda index,A,X,out: index<node_num,\
                  lambda index,A,X,out: [index+1,A,X,maximum_neighborhood(index,A,X,out)],\
                  loop_vars = [tf.zeros([],tf.int32),A,X,output_X],\
                  shape_invariants = [tf.TensorShape([]),A.get_shape(),X.get_shape(),tf.TensorShape([None,node_num,output_dim])])
                  
    output_X.set_shape(output_shape)
    return output_X

def maximum_neighborhood(index,A,X,out):
    neigh = tf.boolean_mask(X,A[:,index])
    max_neigh = tf.reduce_max(neigh,keepdims=True,axis=0)
    out = tf.concat([out,max_neigh],axis=0)
    return out

def _sum_pool(self, X):
    sum_pool = tf.reduce_sum(X*tf.expand_dims(self.real_mask, -1), 1)
    return sum_pool

def _mean_pool(self, X):
    mean_pool = tf.reduce_sum(X*tf.expand_dims(self.real_mask, -1), 1)/tf.reduce_sum(self.real_mask, 1, keepdims=True)
    return mean_pool

def _max_pool(self, X):
    max_pool = tf.reduce_max(X*tf.expand_dims(self.real_mask, -1), 1)
    return max_pool

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

def _fcn_transform_net_rmact(self, model_output, layer_sizes, scope):
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

                if hparams.enable_BN is True:
                    curr_hidden_nn_layer = tf.layers.batch_normalization(
                        curr_hidden_nn_layer,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size

            nn_output = hidden_nn_layers[-1]
            return nn_output


def _fcn_transform_net_rmact_rmb(self, model_output, layer_sizes, scope):
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
                tf.summary.histogram(
                    "nn_part/" + "w_nn_layer" + str(layer_idx), curr_w_nn_layer
                )
                curr_hidden_nn_layer = tf.tensordot(
                        hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                    )

                scope = "nn_part" + str(idx)

                if hparams.enable_BN is True:
                    curr_hidden_nn_layer = tf.layers.batch_normalization(
                        curr_hidden_nn_layer,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size

            nn_output = hidden_nn_layers[-1]
            return nn_output

def _fcn_weighted_net(self, model_output, layer_size, scope):
    """Construct the MLP part for the model.

    Args:
        model_output (obj): The output of upper layers, input of MLP part
        scope (obj): The scope of MLP part

    Returns:s
        obj: prediction logit after fully connected layer
    """
    hparams = self.hparams
    with tf.variable_scope(scope):
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            curr_w_nn_layer = tf.get_variable(
                name="weights" ,
                shape=[layer_size],
                dtype=tf.float32,
            )
            tf.summary.histogram(
                "nn_part/" + "weights", curr_w_nn_layer
            )

            #  curr_hidden_nn_layer = tf.tensordot(
                    #  hidden_nn_layers, curr_w_nn_layer, axes=1
                #  )

            curr_hidden_nn_layer = model_output * tf.expand_dims(tf.expand_dims(curr_w_nn_layer, 0), 0)

            scope = "nn_part" 

            if hparams.enable_BN is True:
                nn_output = tf.layers.batch_normalization(
                    curr_hidden_nn_layer,
                    momentum=0.95,
                    epsilon=0.0001,
                    training=self.is_train_stage,
                )

            return nn_output

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

#  @tf.function
def _gat_mess(self, x, edge_index,
        query_kernel, query_bias, query_activation,
        key_kernel, key_bias, key_activation,
        kernel, bias=None, activation=None, num_heads=1, drop_rate=0.0, training=False):

    num_nodes = tf.shape(x)[0]

    # self-attention
    #  edge_index, edge_weight = add_self_loop_edge(edge_index, num_nodes)
    # 这个函数是给所有节点添加自循环，应改为只给相关节点加

    row, col = edge_index[0], edge_index[1]

    Q = query_activation(x @ query_kernel + query_bias)
    #  Q = x @ query_kernel + query_bias
    #  Q = tf.layers.conv1d(x, 40, 1)
    Q = tf.gather(Q, row)

    K = key_activation(x @ key_kernel + key_bias)
    #  K = x @ key_kernel + key_bias
    #  K = tf.layers.conv1d(x, 40, 1)
    K = tf.gather(K, col)

    V = x @ kernel

    # xxxxx_ denotes the multi-head style stuff
    # 在最后一维切分，在第0维拼接。为了不增加参数量
    Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)
    # 利用节点数更新边索引，在第一维拼接，所以不是最后一维？
    edge_index_ = tf.concat([edge_index + i * num_nodes for i in range(num_heads)], axis=1)

    att_score_ = tf.reduce_sum(Q_ * K_, axis=-1)
    normed_att_score_ = segment_softmax(att_score_, edge_index_[0], num_nodes * num_heads)

    if training and drop_rate > 0.0:
        normed_att_score_ = tf.nn.dropout(normed_att_score_, drop_rate)

    #  h_ = self._aggregate_neighbors(
    h_ = aggregate_neighbors(
        x=V_, edge_index=edge_index_, edge_weight=normed_att_score_,
        #  mapper = self._gcn_mapper,
        mapper = gcn_mapper,
        #  reducer = self._sum_reducer,
        reducer = sum_reducer,
        #  updater = self._identity_updater) # 因为边中有自循环了，所以不再需要加自己
        updater = identity_updater) # 因为边中有自循环了，所以不再需要加自己
        #  updater = self._sum_updater)
        #  updater = sum_updater)

    # 将多头再纵向拆分，横向拼接回来
    h = tf.concat(tf.split(h_, num_heads, axis=0), axis=-1)

    if bias is not None:
        h += bias

    if activation is not None:
        h = activation(h)

    return h

def _mini_batch_gat(self):

    ## graph
    self.n_users = self.user_vocab_length
    self.n_items = self.item_vocab_length
    plain_adj, norm_adj, mean_adj, pre_adj = self.get_adj_mat()
    self.print_statistics(norm_adj, 'normed adj matrix ')
    bias_in = self._convert_sp_mat_to_sp_tensor(norm_adj)

    ## negighbors(sparse version)
    edges = tf.cast(bias_in.indices, tf.int32) # cause involved_items is int32 # (2653785, 2)
    edge_weights = bias_in.values #  (2653785,)
    #  tar_nodes = edges[:,1]
    tar_nodes = edges[:,0] # gat_mess中认为右节点是左节点的邻居，即将右节点聚合到左节点，所以左节点是目标节点

    ## 用自带函数（但是intersection只能找到第一个重复元素的位置）
    #  #  _, involved_index = tf.setdiff1d(edges[1], self.involved_items)
    #  #  involved_edges = tf.gather(edges, involved_index, axis=-1)
    #  #  _, involved_index = tf.sets.difference(edges[1], self.involved_items)
    #  involved_index = tf.sets.intersection(tf.expand_dims(tar_nodes,0),
                                          #  tf.expand_dims(self.involved_items,0)).indices[:,1] # intersection not work in rank1, so use expand_dims
    ## 自己实现，但此为密集矩阵，内存占用不现实
    import pdb; pdb.set_trace()
    #  tar_node_mask = tf.equal(tf.expand_dims(self.involved_items, 0), tf.expand_dims(tar_nodes, 1)) # (1,?) + (2653785,1) -> (2653785,?)
    #  tar_node_mask = tf.equal(tf.expand_dims(self.involved_items, 0), tf.expand_dims(tar_nodes, 1)) # (1,?) + (2653785,1) -> (2653785,?)
    #  tar_node_mask = tf.reduce_max(tf.cast(tar_node_mask, tf.int32), 1) # (2653785)
    tar_node_mask = tf.cast(tf.reduce_any(tf.equal(tf.reshape(edges[:,0], [-1, 1]), self.involved_items), 1), tf.int32)
    involved_index = tf.squeeze(tf.where(tar_node_mask), 1)

    involved_edges = tf.gather(edges, involved_index)
    involved_edge_weights = tf.gather(edge_weights, involved_index)
    #  involved_row = involved_edges[:,0]
    #  involved_col = involved_edges[:,1]

    #  import pdb; pdb.set_trace()

    pp1 = tf.shape(involved_edges)
    pp2 = tf.shape(self.involved_items)
    pp3 = tf.shape(involved_index)
    #  [pp1:][43918 2] # 边数。一个节点就平均连接一条边，可根据统计信息是40....
    #  [pp2:][43918] # 所以问题出在intersection，每个节点只匹配了一条边（这也是mini-batch不如full-batch的原因？）
    #  [pp3:][43918]

    #  pp1 = tf.shape(self.involved_items)
    #  pp2 = tf.shape(self.iterator.items)
    #  pp3 = tf.shape(self.item_history_embedding)
    #  [pp1:][43918] # 目标节点+历史节点数
    #  [pp2:][20000] # 目标节点数，也是下游模型的目标graph数
    #  [pp3:][20000 50 32]

    self.pp1=tf.Print(pp1,["pp1:", pp1]) # 要对应名称，因为执行顺序可能不一样
    self.pp2=tf.Print(pp2,["pp2:", pp2])
    self.pp3=tf.Print(pp3,["pp3:", pp3])

    ## sampled negighbors(sparse version)
    # 注意采样邻居后，把自循环边加回来，再归一化
    # TODO

    ## gat
    #  ego_embeddings = self.item_lookup
    ego_embeddings = self.item_cate_lookup
    all_embeddings = [ego_embeddings]

    for k in range(0, self.n_layers):

        ego_embeddings = self._gat_mess(x = ego_embeddings, 
                  edge_index = tf.transpose(involved_edges), # (?,2) -> (2,?)
                  query_kernel = self.gat_weights['query_kernel_%d' %k], 
                  query_bias = self.gat_weights['query_bias_%d' %k], 
                  query_activation=tf.nn.relu,
                  key_kernel = self.gat_weights['key_kernel_%d' %k], 
                  key_bias = self.gat_weights['key_bias_%d' %k], 
                  key_activation=tf.nn.relu,
                  kernel = self.gat_weights['kernel_%d' %k], 
                  bias = self.gat_weights['bias_%d' %k], 
                  activation=tf.nn.relu, 
                  num_heads=1, drop_rate=0.0, training=self.is_train_stage)

        all_embeddings += [ego_embeddings]
    all_embeddings=tf.stack(all_embeddings,1)
    #  all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
    all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keep_dims=False)

    # last layer for CompositeTensor error
    #  all_embeddings = tf.gather(ego_embeddings + 0, tf.range(0, tf.shape(ego_embeddings)[0]))

    #  cover
    #  self.item_lookup = all_embeddings
    self.item_cate_lookup = all_embeddings

    return all_embeddings
    #  return self.item_lookup

def _gcn(self):
    self.node_dropout_flag = False
    self.n_fold = 100
    self.n_layers = 2
    self.n_users = self.user_vocab_length
    self.n_items = self.item_vocab_length
    plain_adj, norm_adj, mean_adj, pre_adj = self.get_adj_mat()
    self.print_statistics(norm_adj, 'normed adj matrix')
    self.weights = init_gcn_weight()
    self.item_lookup = self._create_lightgcn_embed_ii(norm_adj)
    return self.user_lookup, self.item_lookup

def _create_lightgcn_embed_ii(self, R):
    if self.node_dropout_flag:
        A_fold_hat = self._split_A_hat_node_dropout(R)
    else:
        A_fold_hat = self._split_A_hat(R)

    ego_embeddings = self.item_lookup
    all_embeddings = [ego_embeddings]

    for k in range(0, self.n_layers):

        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

        side_embeddings = tf.concat(temp_embed, 0)

        #  ego_embeddings = side_embeddings
        ego_embeddings = tf.nn.leaky_relu(tf.matmul(side_embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])

        all_embeddings += [ego_embeddings]
    all_embeddings=tf.stack(all_embeddings,1)
    #  all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
    all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keep_dims=False)
    i_g_embeddings = all_embeddings
    return i_g_embeddings

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
    #  ftr_in = self.item_cate_lookup
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
    #  residual = True
    nonlinearity = tf.nn.elu
    #  nonlinearity = tf.nn.leaky_relu

    # gat core
    ftr_out = self.inference(ftr_in, output_dim, nb_nodes, is_train,
                            attn_drop, ffd_drop,
                            bias_mat=bias_in,
                            hid_units=hid_units, n_heads=n_heads,
                            residual=residual, activation=nonlinearity)
    ftr_out = tf.squeeze(ftr_out,0)
    self.item_lookup = ftr_out
    #  self.item_cate_lookup = ftr_out
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

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(self, seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        print('seq', seq)
        #  seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        seq_fts = tf.layers.dense(seq, out_sz, use_bias=False)
        print('seq_fts', seq_fts)

        # simplest self-attention possible
        #  f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_1 = tf.layers.dense(seq_fts, 1)
        #  f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.dense(seq_fts, 1)
        #  f_1 = tf.squeeze(f_1,0)
        #  f_2 = tf.squeeze(f_2,0)
        print('f_1', f_1)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))
        print('f_1', f_1)
        print('adj_mat', adj_mat)

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
                #  ret = ret + tf.layers.conv1d(seq, ret.shape[-1], 1) # activation
                ret = ret + tf.layers.dense(seq, ret.shape[-1]) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

def inference(self, inputs, output_dim, nb_nodes, training, attn_drop, ffd_drop,
        bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
    #  attns = []
    #  for _ in range(n_heads[0]):
        #  attns.append(self.sp_attn_head(inputs,
            #  adj_mat=bias_mat,
            #  out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,
            #  in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
    #  h_1 = tf.concat(attns, axis=-1)
    h_1 = inputs 
    for i in range(0, len(hid_units)):
        attns = []
        for _ in range(n_heads[i]):
            attns.append(self.sp_attn_head(h_1,
                adj_mat=bias_mat,
                out_sz=hid_units[i], activation=activation, nb_nodes=nb_nodes,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
        h_1 = tf.concat(attns, axis=-1)
    #  out = []
    #  for i in range(n_heads[-1]):
        #  out.append(self.sp_attn_head(h_1, adj_mat=bias_mat,
            #  out_sz=output_dim, activation=lambda x: x, nb_nodes=nb_nodes,
            #  in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
    #  logits = tf.add_n(out) / n_heads[-1]

    #  return logits
    return h_1

def _split_A_hat(self, X):
    A_fold_hat = []

    fold_len = X.shape[0] // self.n_fold
    for i_fold in range(self.n_fold):
        start = i_fold * fold_len
        if i_fold == self.n_fold -1:
            end = X.shape[0]
        else:
            end = (i_fold + 1) * fold_len

        A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
    return A_fold_hat

def _split_A_hat_node_dropout(self, X):
    A_fold_hat = []

    fold_len = X.shape[0] // self.n_fold
    for i_fold in range(self.n_fold):
        start = i_fold * fold_len
        if i_fold == self.n_fold -1:
            end = X.shape[0]
        else:
            end = (i_fold + 1) * fold_len

        temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
        n_nonzero_temp = X[start:end].count_nonzero()
        #  A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))
        A_fold_hat.append(self._dropout_sparse(temp, 1-0, n_nonzero_temp))

    return A_fold_hat

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


