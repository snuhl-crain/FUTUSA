
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
from Bio import SeqIO
from goatools import obo_parser
from sklearn.model_selection import train_test_split
from global_setting import *

target_go = f"GO:{target_go_num}"

def fasta_to_df(fasta):
    global max_seq_length
    fasta_sequences = list(SeqIO.parse(fasta,'fasta'))
    temp_entry = []
    temp_seq = []
    temp_len = []
    for i in range(len(fasta_sequences)):
        temp_entry.append(str(fasta_sequences[i].id))
        temp_seq.append(str(fasta_sequences[i].seq))
        temp_len.append(len(str(fasta_sequences[i].seq)))
    fasta_df = pd.DataFrame({'id':temp_entry,'Sequence':temp_seq, 'Length':temp_len})
    fasta_df['Entry'] = fasta_df.id.str.split('|').str[1]
    fasta_df['Name'] = fasta_df.id.str.split('|').str[2]
    del fasta_df["id"]
    fasta_df = fasta_df[fasta_df['Length'] <= max_seq_length]
    fasta_df = fasta_df.reset_index(drop = True)
    return fasta_df
    
def target_labeling(dataset, goa, target_gos):
    temp = []
    ser = list(set(goa[goa['GO'].isin(target_gos)]['Entry']))
    labeled_dataset = dataset.copy()
    labeled_dataset['Target'] = dataset['Entry'].copy().isin(ser)
    return labeled_dataset
    
def get_children(target_go, go_list):
    for i in range(len(go_obo[target_go].children)):
        go_list.append(list(go_obo[target_go].children)[i].id)
        go_list = list(set(go_list))
    return go_list
    
def target_go_listing(target_go):
    target_gos = [target_go]
    temp = 0
    while len(target_gos) != temp:
        temp = len(target_gos)
        for i in range(len(target_gos)):
            target_go = target_gos[i]
            target_gos = get_children(target_go, target_gos)
    return target_gos

print('train/evaluation dataset generating with ' + target_go + '...')
print(f'Maximum sequence length = {max_seq_length}')
go_obo = obo_parser.GODag(GO_path + '/gene_ontology.obo')
goa_human = pd.read_csv(GO_path + '/goa_human_mod.csv')
goa_yeast = pd.read_csv(GO_path + '/goa_yeast_mod.csv')
fasta_human_df = fasta_to_df(FASTA_path + '/fasta_human.fasta')
fasta_yeast_df = fasta_to_df(FASTA_path + '/fasta_yeast.fasta')
fasta_merged = pd.concat([fasta_human_df,fasta_yeast_df])
goa_merged = pd.concat([goa_human,goa_yeast])
if target_protein != None:
    trainset = fasta_merged[fasta_merged['Name']!=target_protein]
    trainset = trainset.reset_index(drop = True)
    evaluset = fasta_merged[fasta_merged['Name']==target_protein]
    evaluset = evaluset.reset_index(drop = True)
    
if random_evaluation == True:
    trainset, evaluset = train_test_split(fasta_merged, test_size=evaluation_ratio, shuffle=True)
target_gos = target_go_listing(target_go)
print('target_GO_terms : ' + str(len(target_gos)))
trainset = target_labeling(trainset, goa_merged, target_gos)
evaluset = target_labeling(evaluset, goa_merged, target_gos)
print('train_target_proteins : ' + str(len(trainset[trainset['Target']==1])))
print('train_total_proteins : ' + str(len(trainset)))
print('evaluation_target_proteins : ' + str(len(evaluset[evaluset['Target']==1])))
print('evaluation_total_proteins : ' + str(len(evaluset)))
trainset.to_csv(dataset_path + f'/{target_go_num}_trainset.csv', index=False)
evaluset.to_csv(dataset_path + f'/{target_go_num}_evaluset.csv', index=False)
