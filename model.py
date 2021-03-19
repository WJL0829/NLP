import tensorflow as tf
from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    # __init__: 为Transformer包含了一个hyperparams的超参数对象，来生成token2idx和idx2token的映射，以及word embedding矩阵
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)
        # get_token_embeddings：生成词向量矩阵，这个矩阵是随机初始化的，且self.embeddings设置成了tf.get_variable共享参数
        # self.embeddings：其维度为（vocab_size，d_model）

    # Transformer模型的编码器部分，
    # 首先是word embedding + position embedding来形成输入数据。
    # 接下来是一个循环num_blocks次的Multi - Head Attention部分。
    # 利用modules.py模块中的multihead_attention()函数来进行搭建，其中causality参数 = False代表只是mask和不是mask for future blinding。
    # multi-head attention之后再接一个Feed Forward的前向网络，由这两个sub_layers构成一个block。
    def encode(self, xs, training=True):  # 实现encode 模型
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        # N：batch_size
        # T1:句子长度
        # d_model：词向量维度
        '''
        # 实现的功能：
        # （1）输入词向量 + positional_encoding
        # （2）encode中共有6个blocks进行连接，每个encode中有multihead attention和全连接层ff进行连接
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # src_masks
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen1)  # 将位置向量添加到初始词向量中
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, sents1, src_masks


    # 解码器部分，首先仍然是一个embedding来构建输入数据的部分，接下来是一个循环num_blocks次的部分。
    # blocks中的第一个是一个Mask Multi - Head Attention，所以需要在multihead_attention()函数中将causality设置为True。
    # 又因为该Attention是self - Attention，所以输入的queries、keys、values都是dec本身。
    # 接下来仍然是一个Multi - Head Attention, 这里的keys、values由encoder的输出提供，所以输入的参数与上面的不同。
    # 这里causality参数设置为False。然后是一个同样的Feed Forward的前向网络
    # 最后是一个decoder解码器的输出部分。这里linear projection的权重矩阵由embedding矩阵的转置得到，
    # 因为最终输出要生成的是一个vocab_size大小的向量，表示输出的各个单词的概率。这里Multi - HeadAttention的输出与权重矩阵的乘法，
    # 没有使用tf.matmul()的矩阵乘法，因为这里两个矩阵的维度大小不相同。
    # Multi - Head Attention的输出维度大小为（N，T2，d_model），而weights的维度大小为（d_model，vocab_size）。
    # 所以这里用了tf.einsum()函数，这里该函数的第一个参数：'ntd,dk->ntk’表示的意思是，->代表乘法操作，ntd,dk->ntk表示两个矩阵相乘后结果的维度为ntk。
    # 这样就实现了两个维度不同矩阵的乘法。
    # liner projuction的输出logits的维度大小为（N，T2，vocab_size）。即代表了在当前T2的长度下，每个位置上的输出各个单词的概率。
    # 然后利用tf.argmax() 在axis = -1的维度下，求出概率最大的那个位置的词作为该位置上的输出单词。所以y_hat的维度大小为（N，T2）。
    def decode(self, ys, memory, src_masks, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # tgt_masks
            tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, self.hp.maxlen2)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # 实现了两个multihead attention结构。
                    # 第一个multihead attention结构和encode模型中的一样，都为self - attention结构
                    # 第二个multihead attention结构，在Q, K, V的输入就不同了，其输入memory实际上是encode模型的输入结果。
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              key_masks=tgt_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        # tf.einsum函数是实现矩阵相乘的方式，用mutual不能进行三维矩阵和二维矩阵的相乘，而einsum则可以，
        # 通过设置参数，如“ntd, dk->ntk”可以得到矩阵维度是[n, t, k]
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y, sents2

    def train(self, xs, ys):  # 用于训练模型
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        # 调用decode（）和encode（）来获取个部分的输出结果
        memory, sents1, src_masks = self.encode(xs)
        logits, preds, y, sents2 = self.decode(ys, memory, src_masks)

        # train scheme
        # 利用one_hot表示每个词的索引在整个词表中的位置, 相当于构建出了要训练的目标Label，
        # 这里就是要使logits的最终结果，即vocab_size大小的向量中，目标词汇所在位置（索引）的值尽可能的大，而使其他位置的值尽可能的小。
        # 构造出了输出和标签之后，就使用tf.nn.softmax_cross_entropy_with_logits()进行训练
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))  # label_smoothing函数用来进行one hot函数的平滑处理
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        # 在计算Loss之前，还要进行一定的处理。由于一开始对有些句子长度不够maxlen的进行了padding，所以在计算Loss的时候，将这些位置上的误差清0
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>
        # loss函数用到了交叉熵函数，但是在计算的时候去掉了padding的影响。
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        #对学习率进行调整，用到了warmup操作，初始阶段lr逐渐上升，迭代后期则逐渐下降
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)
        # 最后即是利用了AdadeltaOptimizer优化器对loss进行优化。
        # tf.summary.scalar()函数即是以key - value的形式保存数值，可以用于TensorBoard中对数据的可视化展示。
        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    # 对模型训练效果进行评估
    # 这里即是让解码器根据开始符 < s > 来自动的进行Machine Translation的过程。流程与test的部分类似，但这里并不是真正的test。
    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)
        # 这里xs[0]表示取第一个tensor x。
        # tf.shape(xs[0]) = [N, T], tf.shape([xs[0])[0]
        # 即取batch_size大小，在evaluation部分，由于解码器的预测仍然是按序列式的进行（与train时候的不同）,
        # 即每一次解码过程预测一个目标词汇，所以在时刻t = 0时解码器的输入维度应该是(N, 1)，
        # 即此时为一个batch输入，每个batch的开头为 < s > 表示开始进行解码，然后每完成一次解码过程，则每个batch已输出词汇数 + 1，
        # 例如t = 1时刻，则解码器的输入维度为(N, 2), 以此类推，直到输入到表示停止。然后再将新的decoder_inputs加入ys中作为下一时刻decoder的输入。
        memory, sents1, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        # 这里这一过程，是不断的进行序列化的预测过程。
        # 循环次数为maxlen2次，表示是要翻译完一整个句子的长度，然后不断的将上一时刻的解码器的输出添加到下一时刻解码器的输入。
        for _ in tqdm(range(self.hp.maxlen2)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)
        # monitor a random sample
        # 随机抽取一个batch来查看模型的结果。n代表从0，batch_size - 1之间选择一个batch sample进行观察。
        # sent1[n]表示原始的输入句子，pred即代表了decoder的预测翻译的输出句子，sents2[n]即表示正确的翻译输出句子。
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries

