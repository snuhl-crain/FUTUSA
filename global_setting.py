
import pandas as pd

FASTA_path = ''
GO_path = ''
model_path = ''
dataset_path = ''
result_path = ''

segmentation_size = 64
target_go_num = "0016491"
aa_letter = 'XARNDCQEGHILKMFPSUTWYV'

max_seq_length = 2000
target_protein = None

random_evaluation = False
evaluation_ratio = 0.05

trainset = pd.read_csv(dataset_path + '')
evaluset = pd.read_csv(dataset_path + '')

train_test_shuffle = True
train_test_shuffle_patience = 3
maximum_train_test_shuffle = 30
early_stopping_patience = 3
continued = False

upper_rank = 10
evaluation = True

best_model_path = model_path + f'FUTUSA_{target_go_num}.hdf5'