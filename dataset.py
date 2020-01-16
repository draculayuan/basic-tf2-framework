import pandas as pd
import numpy as np
import tensorflow as tf
import bert_tokenization as tokenization
from tqdm.notebook import tqdm
from math import floor, ceil

def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(title, question, answer, max_sequence_length, tokenizer,
                t_max_len=30, q_max_len=239, a_max_len=239):

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
            
            
        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" 
                             % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))
        
        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]
    
    return t, q, a

def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        t, q, a = _trim_input(t, q, a, max_sequence_length, tokenizer)

        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])

def construct(root, path, tokenizer_path, MAX_SEQUENCE_LENGTH, test=False, batch_size=1):
    tokenizer = tokenization.FullTokenizer(tokenizer_path, True)
    #MAX_SEQUENCE_LENGTH = 512

    data = pd.read_csv(root+path)
    #df_test = pd.read_csv(PATH+'test.csv')
    #df_sub = pd.read_csv(PATH+'sample_submission.csv')
    #print('test shape =', df_test.shape)
    output_categories = list(data.columns[11:])
    input_categories = list(data.columns[[1,2,5]])
    
    outputs = compute_output_arrays(data, output_categories)
    inputs = compute_input_arays(data, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    #test_inputs = compute_input_arays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    #inputs = [np.expand_dims(x, axis=2) for x in inputs]
    #outputs = np.expand_dims(outputs, axis=2)
    if not test:
        def generator():
            for in1, in2, in3, out in zip(inputs[0], inputs[1], inputs[2], outputs):
                yield {'input_word_ids':in1, 'input_masks': in2, 'input_segments':in3}, out
                #yield [in1, in2, in3], out

        dataset = tf.data.Dataset.from_generator(generator, \
                                                 ({'input_word_ids':tf.int32, 'input_masks': tf.int32, 'input_segments':tf.int32},tf.float32))
        dataset = dataset.batch(batch_size)
        print('Train dataset constructed successfully with shape =', data.shape)
    else:
        # evaluation
        return (inputs, outputs)
    return dataset

def construct_test(root, path, tokenizer_path, MAX_SEQUENCE_LENGTH, test=False, batch_size=1):
    tokenizer = tokenization.FullTokenizer(tokenizer_path, True)
    #MAX_SEQUENCE_LENGTH = 512

    data = pd.read_csv(root+path)
    input_categories = list(data.columns[[1,2,5]])

    inputs = compute_input_arays(data, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

    def generator():
        for in1, in2, in3 in zip(inputs[0], inputs[1], inputs[2]):
            yield {'input_word_ids':in1, 'input_masks': in2, 'input_segments':in3}
            #yield [in1, in2, in3], out

    dataset = tf.data.Dataset.from_generator(generator, \
                                             {'input_word_ids':tf.int32, 'input_masks': tf.int32, 'input_segments':tf.int32})
    dataset = dataset.batch(batch_size)
    print('Test dataset constructed successfully with shape =', data.shape)

    return dataset