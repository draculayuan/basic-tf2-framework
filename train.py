import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow_hub as hub
import tensorflow as tf
import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import gc
import os
from scipy.stats import spearmanr
from math import floor, ceil
import argparse

from model import bert_model
from dataset import construct
from util import CustomCallback

np.set_printoptions(suppress=True)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_and_predict(model, train_dataset, valid_dataset, \
                      learning_rate, epochs, batch_size, loss_function, save_root):
    
    custom_callback = CustomCallback(
        valid_data=(valid_dataset[0], valid_dataset[1]), 
        batch_size=batch_size, save_root=save_root)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit_generator(train_dataset, epochs=epochs, callbacks=[custom_callback])  
              #batch_size=batch_size)
    
    return custom_callback

def main(args):
    #init model
    model = bert_model(args.max_seq_length, args.bert_path)
    print("Bert model loaded")
    #get data
    train_dataset = construct(args.root, args.train_path, args.tokenizer_path, args.max_seq_length, batch_size=args.batch_size)
    valid_dataset = construct(args.root, args.val_path, args.tokenizer_path, args.max_seq_length, test=True)
    #train
    histories = []
    history = train_and_predict(model, 
                          train_dataset, 
                          valid_dataset,
                          #test_data=test_inputs, 
                          learning_rate=3e-5, epochs=4,
                          loss_function='binary_crossentropy',
                          batch_size=args.batch_size,
                          save_root=args.save_root)

    histories.append(history)
    # save model
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tensorflow NLP framework")
    parser.add_argument('--max_seq_length', default=512)
    parser.add_argument('--root', default='/home/liu/DL_workstation/Google_QA/data/')
    parser.add_argument('--bert_path', default='/home/liu/DL_workstation/Google_QA/code/bert')
    parser.add_argument('--tokenizer_path', default='/home/liu/DL_workstation/Google_QA/code/bert/assets/vocab.txt')
    parser.add_argument('--train_path', default='train.csv')
    parser.add_argument('--val_path', default='dev_val.csv')
    parser.add_argument('--save_root', default='/home/liu/DL_workstation/Google_QA/code/ckpt/')
    parser.add_argument('--batch_size', default=2)
    
    args = parser.parse_args()
    main(args)
    
    