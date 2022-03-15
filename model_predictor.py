
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, Sequence
import keras.backend as K
from global_setting import *
from model_FUTUSA import *

target_go = f"GO:{target_go_num}"

aa_to_int = dict((c, i) for i, c in enumerate(aa_letter))
int_to_aa = dict((i, c) for i, c in enumerate(aa_letter))


def predict_function(upper_rank, result_path):
    Score_mean = []
    Score_rank = []
    
    for i in predset.Sequence[:]:
        print(i)
        pred_input = []
        pred_input_temp = []
        pred_result_temp = []
        temp_padding = (segmentation_size-1)*'X' + i + (segmentation_size)*'X'
    
        for j in range(len(temp_padding)+1):
            if j >= segmentation_size:
                pred_input_temp.append(temp_padding[j-segmentation_size:j])
        for k in pred_input_temp:
            temp_aa = []
            for w in k:
                temp_aa.append(aa_to_int[w])
            pred_input.append(temp_aa)
        pred_input = pad_sequences(pred_input, maxlen=segmentation_size, padding='post')
        pred_input = to_categorical(pred_input,num_classes=len(aa_to_int))
        for l in range(len(pred_input)-1):
            if l > segmentation_size+1:
                prediction = model.predict_on_batch(pred_input[l].reshape(1,segmentation_size,len(aa_to_int)))
                K.clear_session()
                pred_result_temp.append(prediction[0][0])
        Score_mean.append(np.mean(np.array(pred_result_temp)))
        Score_rank.append(np.mean(np.array([pred_result_temp[i] for i in np.argsort(pred_result_temp)[-len(pred_result_temp)//upper_rank:]])))
        print(np.mean(np.array(pred_result_temp)))
        print(np.mean(np.array([pred_result_temp[i] for i in np.argsort(pred_result_temp)[-len(pred_result_temp)//upper_rank:]])))
    predset['Score_mean'] = Score_mean
    predset['Score_rank'] = Score_rank
    predset.to_csv(result_path + f'{target_go_num}_prediction_result.csv')
    return
  
if evaluation == True:
    predset = pd.read_csv(dataset_path + f'{target_go_num}_evaluset.csv')
else:
    predset = pd.read_csv(dataset_path + f'{target_go_num}_predset.csv')
K.clear_session()
model = model_FUTUSA(segmentation_size, aa_letter)
model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['binary_accuracy','Recall'])
model.summary()
    
model.load_weights(best_model_path)

predict_function(upper_rank, result_path)