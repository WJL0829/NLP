import tensorflow as tf
from utils import calc_num_batches
# 加载词汇表, 根据已构建的词表来构建idx2token和token2idx两个映射字典。
# vocab_fpath: 词文件的地址
# 0: <pad>, 1: <unk>, 2: <s>, 3: </s>
def load_vocab(vocab_fpath):
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    #返回两个字典，一个是id->token，一个是token->id
    return token2idx, idx2token

# load_data: 分别将中文和英语的句子读到sents1列表和sents2列表,当句子超过长度maxlen1或者maxlen2的时候就删除,以列表的形式进行保存每个满足条件的sentence.
def load_data(fpath1, fpath2, maxlen1, maxlen2):
    sents1, sents2 = [], []
    with open(fpath1, 'r') as f1, open(fpath2, 'r') as f2:
        for sent1, sent2 in zip(f1, f2):
            # 长度超过阈值的丢弃
            if len(sent1.split()) + 1 > maxlen1: continue # 1: </s>
            if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
            sents1.append(sent1.strip())
            sents2.append(sent2.strip())
    # 保存句子，返回输入和输入句子的列表
    return sents1, sents2

# encode:函数根据idx2token和token2idx映射表，将每个单词转换成对应的idx表示
# 对encode部分输入的sentence末尾加入了</s>结束符，
# 在decode部分的sentence开头加入<s>表示解码开始，</s>表示解码结束。
def encode(inp, type, dict):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    inp_str = inp.decode("utf-8")
    # 在encoder输入句子x中，添加了结尾符号“</s>”
    # 在输出句子y中，添加了开头符号“<s>”和结尾符号“</s>”
    if type=="x": tokens = inp_str.split() + ["</s>"]
    else: tokens = ["<s>"] + inp_str.split() + ["</s>"]
    # 返回指定键值，否则返回默认值<unk>
    x = [dict.get(t, dict["<unk>"]) for t in tokens]
    return x


# generator_fn: source侧返回（转换后的句子编码，句子编码长度，原始句子）
#               target侧返回（decoder的输入，转换后的句子编码，句子编码长度，原始句子）
#               这里注意decoder_input,是不包含最后一个字符</s>的句子编码（例如：<s>23,56,.....）,通过这里看出decoder输入的第一个字符是<s>。
# 对于每一个sent1，sent2（源句子，目标句子），sent1经过前面的encode函数转化成x，sent2经过前面的encode函数转化成y之后，
# decoder的输入decoder_input是y[:-1]，预期输出y是y[1:] 即用来解码输入的前N-1个，期望的输出是从第2个到第N个，也是N-1个。
def generator_fn(sents1, sents2, vocab_fpath):
    '''Generates training / evaluation data 生成训练和评估集数据
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        x: list of source token ids in a sent  # 由Idx表示的sentence，维度大小为(N,T1)
        x_seqlen: int. sequence length of x  # 表示句子的长度，维度为(N，)
        sent1: str. raw source (=input) sentence  # 表示原始由token表示的sentence，维度为（N，）
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs  # 不包括结束符</s>
        y: list of target token ids in a sent  # 不包括开始符<s>
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    token2idx, _ = load_vocab(vocab_fpath)
    for sent1, sent2 in zip(sents1, sents2):
        x = encode(sent1, "x", token2idx)
        y = encode(sent2, "y", token2idx)
        decoder_input, y = y[:-1], y[1:]

        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2)

# 用于生成batch数据
# 该方法产生batch，是先repeat()之后，再产生batch数据的，
# 这样会造成最后一个batch如果长度小于batch_size，那么最后几条数据是之前batch里会出现过的，可能会影响到loss的评估
# 因此模型中计算loss时是把所有非padding的部分的交叉熵保留了下来，加起来，除以非padding序列的长度，但是并没有除以batch_size，
# 也就是算的是一个batch里面的总loss，也就对应了他先repeat()再产生batch数据，也就是每个batch中数据的条目数是相等的，这样就会造成训练集和验证集的loss是有问题的
# 但是测试集并不是用loss来衡量的，而是用bleu值。因此如果按照这样的方法产生batch数据，
# 测试集合比如说有900条数据，batch size=128，那么测试集会生成1024条数据，但是代码中他取了前900条数据，先写入生成结果，然后计算bleu值，就不会出现问题。
def input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=False):  #
    '''Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)  # 句子中每个词语转换为id
        x_seqlens: int32 tensor. (N,)  # 句子原有长度
        sents1: str tensor. (N,)  # 单个句子
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)  # 句子中每个词语转换为id，decoder输入
        y: int32 tensor. (N, T2)  # 句子中每个词语转换为id，decoder输出
        y_seqlen: int32 tensor. (N, )  # 句子原有长度
        sents2: str tensor. (N,)  # 单个句子
    '''
    shapes = (([None], (), ()),
              ([None], [None], (), ())) # 与generator返回值对应
    types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.int32, tf.string))  # 与上边的shapes对应
    paddings = ((0, 0, ''),
                (0, 0, 0, ''))

    # 用“tf.data.Dataset.from_generator”加载数据，返回一个生成器对象，生成器方法为：generator_fn
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    # 用"0" padding 句子，把所有的词语转换成对应的词典id
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    # 返回的dataset中，有两个变量xs和ys。
    # xs中包含: x，x_seqlens，sents1
    # ys中包含：decoder_input，y，y_seqlen，sents2
    return dataset

# get_batch:用于生成多个batch数据
# 主要使用了tf.data.Dataset.from_generator加载数据的方法，把句子中的词语生成在词典中的id
def get_batch(fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle=False):
    '''获取training / evaluation mini-batches
    fpath1: 源文件路径 string.
    fpath2: 目标文件路径 string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean
    '''
    sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2)  # 利用input_fn()函数返回数据集生成器对象batches
    batches = input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)  # 根据样本总数和batch_size的大小计算出所需要的batch数目num_batches，以及样本总数len(sents1)。
    return batches, num_batches, len(sents1)
    # batches：tf.data.Dataset的一种形式，包含了元组xs（）和元组ys
    # num_batches: number of mini-batches  总共有多个个batches进行迭代，也就是有多少轮epoch
    # len(sents1):数据集的大小



