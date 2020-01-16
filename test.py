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
from dataset import construct_test

np.set_printoptions(suppress=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def inference(model, test_dataset, batch_size, root, sub_path, result_path):
    
    predictions = model.predict(test_dataset)
    df_sub = pd.read_csv(root+sub_path)
    df_sub.iloc[:, 1:] = np.array(predictions)
    
    df_sub.to_csv(result_path, index=False)
    

def main(args):
    #init model
    model = bert_model(args.max_seq_length, args.bert_path)
    model.load_weights(args.ckpt_path).expect_partial()
    print("Trained model loaded")
    #get data
    test_dataset = construct_test(args.root, args.test_path, args.tokenizer_path, args.max_seq_length, test=True)
    #train
    histories = []
    history = inference(model, 
                        test_dataset, 
                        args.batch_size,
                        args.root,
                        args.sub_path,
                        result_path=args.result_path)

    histories.append(history)
    # save model
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tensorflow NLP framework")
    parser.add_argument('--max_seq_length', default=512)
    parser.add_argument('--root', default='/home/liu/DL_workstation/Google_QA/data/')
    parser.add_argument('--bert_path', default='/home/liu/DL_workstation/Google_QA/code/bert')
    parser.add_argument('--ckpt_path', default='/home/liu/DL_workstation/Google_QA/code/ckpt/bert-base-3.h5py')
    parser.add_argument('--tokenizer_path', default='/home/liu/DL_workstation/Google_QA/code/bert/assets/vocab.txt')
    parser.add_argument('--test_path', default='test.csv')
    parser.add_argument('--sub_path', default='sample_submission.csv')
    parser.add_argument('--result_path', default='/home/liu/DL_workstation/Google_QA/code/inference/submission.csv')
    parser.add_argument('--batch_size', default=2)
    
    args = parser.parse_args()
    main(args)
    
    