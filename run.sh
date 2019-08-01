#!/bin/bash
export  THEANO_FLAGS=mode=FAST_RUN,device=cpu,optimizer=None
python code/run_qa_model.py --queries_folder ../APES --train_path cnn.txt --dev_path cnn_test.txt --glove_path /media/backup1/glove/glove.6B.100d.txt
