# CNN/Daily Mail Reading Comprehension Task

This is a fork of fork of https://github.com/danqi/rc-cnn-dailymail. code for [A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task](https://arxiv.org/pdf/1606.02858v2.pdf).

Issues of this and the original repo:

* Original links of dataset to create word dictionary is dead(vital to recreate this work)

    * no preprocess code was provide to construct the mono text file used to train the model (I have provide the code in this repo)

* for every new line I read the code makes me cringe so bad 

    * Example : load_data function returns an array : documents, questions, answer. The original author however, decided it's nice to name this mix of confusion array as 'train_examples'. see main.py line 153


~~For explaination go to the [original repository](https://github.com/danqi/rc-cnn-dailymail).~~

This fork is to enable quering QA on a trained model. A trained model over the [CNN dataset](http://cs.stanford.edu/~danqi/data/cnn.tar.gz) is available [here](https://github.com/mataney/rc-cnn-dailymail/blob/master/code/model.pkl.gz).

## Dependencies
* Python 2.7
* Theano >= 0.7
* Lasagne 0.2.dev1

## Train model
For more explanation about training a model go to the [original repository](https://github.com/danqi/rc-cnn-dailymail).

## Running

1. Since the preprocessed data is not available, you have to prep CNN/Dailymail questions dataset

```
    (env) python -m code.prep --folder_path=/media/backup1/cnn_dailymail/cnn --output=cnn.txt
```

2.  When running `python code/run_qa_model.py --queries_folder folder_path ..` you start a QA stream that expects questions in `folder_path/queries.pkl` and returns its rewarding accuracy in `folder_path/rewards.txt`

```
    (env) python code/run_qa_model.py --queries_folder ../APES --train_path cnn.txt --dev_path cnn_test.txt --glove_path /media/backup1/glove/glove.6B.100d.txt
```

You should see output like this:

```
...
08-01 23:26 <lasagne.layers.embedding.EmbeddingLayer object at 0x7fb9f252d490>
08-01 23:26 <lasagne.layers.noise.DropoutLayer object at 0x7fb9f2503e90>
08-01 23:26 <lasagne.layers.input.InputLayer object at 0x7fb9f252d450>
08-01 23:26 <lasagne.layers.recurrent.GRULayer object at 0x7fb9f2503f10>
08-01 23:26 <lasagne.layers.noise.DropoutLayer object at 0x7fb9f268ded0>
08-01 23:26 <lasagne.layers.recurrent.GRULayer object at 0x7fb9f2358310>
08-01 23:26 <lasagne.layers.merge.ConcatLayer object at 0x7fb9f2503d10>
08-01 23:26 <nn_layers.BilinearAttentionLayer object at 0x7fb9f27041d0>
08-01 23:26 <lasagne.layers.dense.DenseLayer object at 0x7fb9f2704350>
08-01 23:27 Done.
*****************************Started answering to questions****************************

```
