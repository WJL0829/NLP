import numpy as np
import tensorflow as tf
# 实现layer normalizaiton，即需要计算的是在某一层上的mean和variance，LN的操作类似于将BN做了一个“转置”，对同一层网络的输出做一个标准化。
def ln(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization.
    inputs: 一个有2个或更多维度的张量，第一个维度是batch_size
    epsilon: A floating number. 很小的数值，防止区域划分错误
    scope: Optional scope for `variable_scope`.
      
    Returns: 返回一个与inputs相同shape和数据的dtype
    '''
    # 使用层归一layer normalization
    # tensorflow在实现Batch Normalization（各个网络层输出的归一化）时，主要用到nn.moments和batch_normalization
    # 其中moments作用是统计矩，mean是一阶矩，variance则是二阶中心矩
    # tf.nn.moments计算返回的mean和variance作为tf.nn.batch_normalization参数进一步调用
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    # 初始化word embedding，用矩阵表示，目前为[vocab_size, num_units]的矩阵，索引为0的列设置为0
    # vocab_size为词的数量，num_units为embedding size，一般根据论文设置为512
    '''Constructs token embedding matrix.
    vocab_size: scalar. V.
    num_units: 嵌入维数. E.
    zero_pad: Boolean. 如果为True，则第一行的所有值（id = 0）应为常数零
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.contrib.layers.xavier_initializer())  # 对embedding的数值进行初始化
        if zero_pad:  # zero_pad参数是为了使queries / keys的mask更加方便
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings

# 即论文里提到的Encoder-Decoder Attention，是两个不同序列之间的attention，与来源于自身的 self-attention 相区别。
# context-attention有很多，这里使用的是scaled dot-product。通过 query 和 key 的相似性程度来确定 value 的权重分布。
# 这部分代码是self attention用到的QKV的公式的核心代码, 不管是Encoder-Decoder Attention还是Self Attention都是用的这里的scaled dot-product方法。
def scaled_dot_product_attention(Q, K, V, key_masks,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    查看原论文中3.2.1attention计算公式：Attention(Q,K,V)=softmax(Q K^T /√dk ) V
         :param Q: 查询，三维张量，[N, T_q, d_k].
         :param K: keys值，三维张量，[N, T_k, d_v].
         :param V: values值，三维张量，[N, T_k, d_v].
         :param causality: 布尔值，如果为True，就会对未来的数值进行遮盖
         :param dropout_rate: 0到1之间的一个数值
         :param training: 布尔值，用来控制dropout
         :param scope:`variable_scope`的可选范围
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product，transpose为了使维度能够匹配，得到一个N，T_q, T_q的tensor
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs

# padding mask 在所有的 scaled dot-product attention 里面都需要用到
# sequence mask 只有在 decoder 的 self-attention 里面用到。
def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"

    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],

       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):  # padding mask
        # 因为Q乘以V，V的序列后面有很长一部分是全零的向量（即自定义的padding的对应embedding，定义为全0）
        # 因此全零的部分让attention的权重为一个很小的值-4.2949673e+0
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]) # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
    elif type in ("f", "future", "right"):
        # 是decoder的self attention时要用到的sequence mask，
        # 由于在解码器的部分，不像编码器一样可以直接并行运行，解码器由于需要翻译，仍然是序列化的进行。
        # 所以在decoder的Attention部分，某一时刻翻译出来的词只能跟它之前翻译出的词产生联系。
        # 即只能跟它之前的词计算Attention score.所以这里就用了一个下三角矩阵。
        # 这样在计算Attention时，前面的word不会与在它后面的word产生联系。
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs

# 多头self attention就是Transoformer的核心，用上面提到的QKV公式算出分布之后，把h份合在一起来表示
# 这部分代码主要是先产生QKV向量，然后按照h头来进行划分，然后调用上面的scaled dot-product的方法来计算的。
# 这里将8份self attention分别计算后后concat起来了，然后在self attention层后接了残差连接和layer normalization。
# 对于self attention来说，Q=K=V，而对于decoder-encoder attention来说，Q=decoder_input，K=V=memory。
def multihead_attention(queries, keys, values, key_masks,
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.  头的数量
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.  # dropout机制的控制器
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''

    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        # 三个tf.layers.dense()相当于与三个矩阵（Wq, Wk, Wv）相乘。
        # 经过一个线性映射将queries、keys、values从d_k，d_v的维度映射到d_model的维度，
        # 接下来这里的多头注意力机制并不是分开的8个tensor，而是对原始的Q、K、V矩阵进行切分在进行拼接而成的。
        Q = tf.layers.dense(queries, d_model, use_bias=True) # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True) # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True) # (N, T_k, d_model)
        
        # Split and concat
        # 首先对于d_model = 512，由于采用了num_heads = 8，
        # 所以一开始tf.split()函数的axis = 2，即沿d_model维度(最后一个维度)进行切分，切分成8片。
        # 然后对每片，沿第一个维度batch_size的维度进行拼接，即形成了维度为（h * N, T_q, d_model / h）的维度（针对矩阵Q而言）。
        # 该维度可以明显的看出，生成了h个大小为(N，T_q，d_model / h)的矩阵Q_
        # Q_就包含了所有multihead attention要用到的q, 将它们堆叠到一块了。
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

        # Attention
        # 在对这h个Q_、K_、V_矩阵做scaled_dot_product_attention之后，再进行Reshape的操作，即做跟以上切分相反的操作：
        # 先按axis = 0 第一个维度做切分，相当于生成h个维度大小为(N，T_q，d_model / h)的矩阵，
        # 然后再对这h个矩阵按axis = 2, 即按（d_model / h）的维度进行拼接，从而重新生成大小为（N，T_q，d_model）的矩阵。
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = ln(outputs)
 
    return outputs

def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3  # 实现全连接的前馈神经网络的部分
    
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    # 前向网络是两层全连接层接一个残差连接和layer normalization。　
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        # 第一个代表的从输入层到第一隐层的过程，所以num_hiddens设置为num_units[0]
        # 利用了tf.layers.dense()函数来实现全连接的网络
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        # 第二个则是由第一隐层到输出层，所以num_hiddens设置为num_units[1]
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = ln(outputs)
    
    return outputs

# Label Smoothing：就相当于是使矩阵中的数进行平滑处理。把0改成一个很小的数，把1改成一个比较接近于1的数
# 论文中说这虽然会使模型的学习更加不确定性，但是提高了准确率和BLEU score
def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)

# 由于模型没有循环和卷积，为了让模型知道句子的编号，就必须加入某些绝对位置信息，来表示token之间的关系。
# 就目前而言，Transformer 架构还没有提取序列顺序的信息，如果缺失了这个信息，可能出现：所有词语都对了，但是无法组成有意义的语句。
# 因此模型对序列中的词语出现的位置进行编码。论文中使用的方法是在偶数位置使用正弦编码，在奇数位置使用余弦编码。
# positional_encoding:生成一个位置embedding, 句子中的单词根据位置查这个embedding.
def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"): # positional encoding和embedding有相同的维度，这两个能够相加。

    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # 将余弦应用于偶数列，将sin应用于奇数。
        # position_enc[:，0::2]中的0::2 代表的意义即是从第一个元素开始(0)，每隔2个位置进行遍历，即取了下标为偶数的位置。
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
        # tf.where(tensor, a, b): a, b为和tensor相同维度的tensor，将tensor中的True位置元素替换为ａ中对应位置元素，False的替换为ｂ中对应位置元素

        return tf.to_float(outputs)

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay  # 学习率衰减
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
