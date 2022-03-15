from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, Sequence
import keras.backend as K
from global_setting import *
from model_FUTUSA import *

aa_to_int = dict((c, i) for i, c in enumerate(aa_letter))
int_to_aa = dict((i, c) for i, c in enumerate(aa_letter))

model = model_FUTUSA(segmentation_size, aa_letter)
trainset = pd.read_csv(dataset_path + f'{target_go_num}_trainset.csv')
evaluset = pd.read_csv(dataset_path + f'{target_go_num}_evaluset.csv')

def pred_score(seq, segmentation_size, ID):
    temp_padding = (segmentation_size-1)*'X' + seq + (segmentation_size)*'X'
    Score = []
    aa = []
    pred_input = []
    pred_input_temp = []
    pred_result = []
    pred_result_temp = []
    for j in range(len(temp_padding)):
        if j >= segmentation_size:
            pred_input_temp.append(temp_padding[j-segmentation_size:j])
    for k in pred_input_temp:
        temp_aa = []
        for w in k:
            temp_aa.append(aa_to_int[w])
        pred_input.append(temp_aa)
    pred_input = pad_sequences(pred_input, maxlen=segmentation_size, padding='post')
    pred_input = to_categorical(pred_input,num_classes=len(aa_to_int))
    for l in range(len(pred_input)):
        prediction = model.predict_on_batch(pred_input[l].reshape(1,segmentation_size,len(aa_to_int)))
        K.clear_session()
        pred_result_temp.append(np.array(prediction)[0][0])
    for i in range(len(seq)):
        aa.append(str(seq[i])+str(i+1))
        pred_result.append(np.mean(pred_result_temp[i:i+segmentation_size]))
    df = pd.DataFrame(data={'seq':aa,'score':pred_result})
    plt.scatter(x=range(len(pred_result)),y=pred_result)
    plt.xlabel('Position')
    plt.ylabel('Score')
    plt.ylim(0,1)
    plt.title(str(ID))
    plt.savefig('', dpi=600, bbox_inches='tight')
    return pred_result_temp, df

def pred_pmutscore(seq, pred_origin, segmentation_size, date, model_num, target_protein, continued=pd.DataFrame(columns=['seq','score']), se7=False):
    start = time.time()
    aa_letter = 'XARNDCQEGHILKMFPSUTWYV'
    if len(continued.columns) != 2:
        pred_origin = pd.DataFrame.copy(continued)
        del continued['seq']
        del continued['score']
        col = list(continued.columns) 
        col= ''.join(col)
        aa_letter =  aa_letter.replace(col,'')
    temp_padding = (segmentation_size-1)*'X' + seq + (segmentation_size)*'X'
    for mut_aa in aa_letter[:]:
        print(mut_aa)
        pred_mut = []  
        for aa_index in range(len(seq)):
            pred_input = []
            pred_input_temp = []
            pred_result = []
            pred_result_temp = []
            if mut_aa == 'X':
                mutseq = temp_padding[:aa_index+ segmentation_size-1] + '' + temp_padding[aa_index+segmentation_size:]
            else:
                mutseq = temp_padding[:aa_index+ segmentation_size-1] + mut_aa + temp_padding[aa_index+segmentation_size:]
            for j in range(len(mutseq)):
                if j >= segmentation_size:
                    pred_input_temp.append(mutseq[j-segmentation_size:j])
            for k in pred_input_temp[aa_index:aa_index+segmentation_size]:
                temp_aa = []
                for w in k:
                    temp_aa.append(aa_to_int[w])
                pred_input.append(temp_aa)
            pred_input = pad_sequences(pred_input, maxlen=segmentation_size, padding='post')
            pred_input = to_categorical(pred_input,num_classes=len(aa_to_int))
            for l in range(len(pred_input)):
                prediction = model.predict_on_batch(pred_input[l].reshape(1,segmentation_size,len(aa_to_int)))
                K.clear_session()
                pred_result_temp.append(np.array(prediction)[0][0])
            pred_mut.append(np.mean(pred_result_temp) - pred_origin['score'][aa_index])
        if mut_aa == 'X':
            pred_origin["del"] = pred_mut
        else:
            pred_origin[str(mut_aa)] = pred_mut
        pred_origin.to_csv('',index=False)
        print("duration :", (time.time() - start)/60, "min")
    pred_origin.to_csv('',index=False)
    if se7 == True:
        heat = pred_origin.T.rename(columns=pred_origin.T.iloc[0]).drop(['seq','score'])
    else:
        heat = pred_origin.T.rename(columns=pred_origin.T.iloc[0]).drop(['seq','score','U'])
    heat = heat.astype(dtype='float32')

    plt.figure(figsize=(30, 10))
    ax = sns.heatmap(heat,cmap='coolwarm',  center = 0,  yticklabels=True)
    plt.title(str(target_protein), fontsize=20)
    plt.savefig('', dpi=600, bbox_inches='tight')
    return pred_origin

seq = evaluset[evaluset['Target']==1]['Sequence'].reset_index(drop=True)[0]

a, b  = pred_score(seq, segmentation_size, target_protein)
b = pred_pmutscore(seq, b, segmentation_size, date, model_num,target_protein,continued=cont)