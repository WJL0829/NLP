# 机器翻译
**用Transformer实现 中-->英 翻译系统**    

参考代码：
> *  https://github.com/Kyubyong/transformer 

## 1、运行环境
                                      
* Ubuntu 16.04.7  

* python 3.6  

* tensorflow 1.12-gpu

## 2、代码结构 

* hparams.py：包含实验所需要的所有参数

* prepro.py：加载原始数据并进行预处理，生成源语言和目标语言的词汇文件

* data_load.py：用于构建数据集、对数据集根据词表进行转换等功能

* model.py ：利用已经构建的各组件，来构建整体的Transformer模型结构。

* modules.py ：用于构建Transformer模型中的各部分的模型结构

* train.py ：训练模型，定义了模型、损失函数以及训练和保存模型的过程，包含了评估模型的效果

* test.py ：生成测试文件

* utils.py：调用到的工具类代码，起到一些协助的作用
## 3、代码说明
因为代码来自github，所以我在代码中进行了详细注释。

主要涉及到的改动包括：  

（1）源代码中批大小定义为128，在运行时报错：  
> tensorflow.python.framework.errors_impl.ResourceExhaustedError   
       
因此将批大小调整为64 

（2）根据老师提供的数据集的格式对数据预处理的方式进行了修改    

（3）原代码使用multi-bleu-detok.perl计算bleu分数。本次作业要求使用sacrebleu，因此对测试部分进行了增删



## 4、参数设置
在hparams.py中进行了参数设置
* batch_size = 64
* eval_batch_size = 32
* lr = 0.0003 
* warmup_steps = 4000
* num_epochs = 20  
……

## 5、测试命令
(1)  创建预处理的 train data、test data、eval data
```
python prepro.py
```
(2) 训练模型
```
python train.py
```
(3) 测试模型

```
python test.py --ckpt log/1/
```
训练结果都保存在./log/1/中，运行test.py将下列文件合并后存储于./test/1/:
>iwslt2017_E20L3.29-73180.data-00000-of-00001     
>iwslt2017_E20L3.29-73180.index  
>iwslt2017_E20L3.29-73180.meta

生成测试文件iwslt2017_E20L3.29-73180

(4) sacrebleu测试
```
cat ./test/1/iwslt2017_E20L3.29-73180 | sacrebleu ./data/test.en.tok
```

## 6、运行结果
![avatar](./result.PNG)  

改进方向（未实现）：模型中利用BPE进行中文分词，效果并不是很好，可以改变分词方法



