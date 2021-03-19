import os
import errno
import sentencepiece as spm
import re
from hparams import Hparams
import logging

logging.basicConfig(level=logging.INFO)

def prepro(hp):

    logging.info("# Check if raw files exist")  # 原始数据文件的位置
    train1 = "./data/train.zh.tok"
    train2 = "./data/train.en.tok"
    eval1 = "./data/valid.zh.tok"
    eval2 = "./data/valid.en.tok"
    test1 = "./data/test.zh.tok"
    test2 = "./data/test.en.tok"
  
    logging.info("# Preprocessing")
    # train
    # 对数据进行预处理，除去以" <" 和 "&lt"开头的行
    _prepro1 = lambda x: [line.strip() for line in open(x, 'r').read().split("\n") \
                      if not line.startswith("  <")]
    prepro_train1 = _prepro1(train1)
    _prepro2 = lambda x: [line.strip() for line in open(x, 'r').read().split("\n") \
                      if not line.startswith("&lt")]

    prepro_train2 = _prepro2(train2)
    # 断言，由于是对齐后的语料库，所以在划分之后需要确定prepro_train1和prepro_train2每一行的单词是一致的
    assert len(prepro_train1) == len(prepro_train2), "Check if train source and target files match."

    # eval
    # 用换行符来区分每一行，并除去首尾多余的空格
    _prepro = lambda x: [line.strip() for line in open(x, 'r').read().split("\n")]
    prepro_eval1, prepro_eval2 = _prepro(eval1), _prepro(eval2)
    # 判断两个文件内容长度是否匹配
    assert len(prepro_eval1) == len(prepro_eval2), "Check if eval source and target files match."

    # test
    # 与eval处理一致
    prepro_test1, prepro_test2 = _prepro(test1), _prepro(test2)
    assert len(prepro_test1) == len(prepro_test2), "Check if test source and target files match."

    logging.info("# write preprocessed files to disk")
    os.makedirs("prepro", exist_ok=True)
    def _write(sents, fname):
    # 定义函数，便于将处理过的数据写入文件
        with open(fname, 'w') as fout:
            fout.write("\n".join(sents))

    _write(prepro_train1, "prepro/train.zh")
    _write(prepro_train2, "prepro/train.en")
    _write(prepro_train1+prepro_train2, "prepro/train")
    _write(prepro_eval1, "prepro/eval.zh")
    _write(prepro_eval2, "prepro/eval.en")
    _write(prepro_test1, "prepro/test.zh")
    _write(prepro_test2, "prepro/test.en")

    # 采用Byte pair encoding(BPE)算法进行分词
    # BPE通过一个固定大小的词汇表来表示开放词汇，这个词汇表里面的是变长的字符串序列。这是一种神经网络模型的词分割策略
    logging.info("# Train a joint BPE model with sentencepiece")
    os.makedirs("segmented", exist_ok=True)
    # 输入训练文件 输出文件的前缀名称
    train = '--input=prepro/train --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=segmented/bpe --vocab_size={} \
             --model_type=bpe'.format(hp.vocab_size)
    spm.SentencePieceTrainer.Train(train)

    logging.info("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()
    sp.Load("segmented/bpe.model")

    logging.info("# Segment")
    def _segment_and_write(sents, fname):
        with open(fname, "w") as fout:
            for sent in sents:
                pieces = sp.EncodeAsPieces(sent)  # 对文件进行编码
                fout.write(" ".join(pieces) + "\n")

    _segment_and_write(prepro_train1, "segmented/train.zh.bpe")
    _segment_and_write(prepro_train2, "segmented/train.en.bpe")
    _segment_and_write(prepro_eval1, "segmented/eval.zh.bpe")
    _segment_and_write(prepro_eval2, "segmented/eval.en.bpe")
    _segment_and_write(prepro_test1, "segmented/test.zh.bpe")

if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
    logging.info("Done")