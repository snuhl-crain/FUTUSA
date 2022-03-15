
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import *
import keras.backend as K
from sklearn.model_selection import train_test_split
import multiprocessing
import tracemalloc
from global_setting import *
from model_FUTUSA import *

target_go = f"GO:{target_go_num}"
trainset = pd.read_csv(dataset_path + f'{target_go_num}_trainset.csv')
evaluset = pd.read_csv(dataset_path + f'{target_go_num}_evaluset.csv')


aa_to_int = dict((c, i) for i, c in enumerate(aa_letter))
int_to_aa = dict((i, c) for i, c in enumerate(aa_letter))

class DataGenerator(Sequence):
    def __init__(self, x, y, batch_size=256):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.x))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __data_generation(self, x_batch, y_batch):
        x_out = np.zeros((self.batch_size, segmentation_size, len(aa_to_int)), dtype=np.int32)
        y_out = np.zeros((self.batch_size, 1), dtype=np.int32)
        x_out = pad_sequences(x_batch, maxlen=segmentation_size, padding='post')
        x_out = to_categorical(x_out,num_classes=len(aa_to_int))
        y_out = np.array(y_batch).astype(np.int32)

        return x_out, y_out


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        x_temp = [self.x[k] for k in indexes]
        y_temp = [self.y[k] for k in indexes]
        x_batch, y_batch = self.__data_generation(x_temp, y_temp)

        return x_batch, y_batch
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

def input_gen(dataset):
    global segmentation_size
    data_input = []
    data_target = []
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"data_input start Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    for i in range(len(dataset)):
        temp_padding = (segmentation_size-1)*'X' + dataset['Sequence'][i] + (segmentation_size)*'X'
        data_input_temp = []
        for j in range(0,len(temp_padding)):
            if j > segmentation_size:
                data_input_temp.append(temp_padding[j-segmentation_size:j])
                if dataset['Target'][i] == True:
                    data_target.append('1')
                else:
                    data_target.append('0')
        for k in data_input_temp:
            temp_aa = []
            for w in k:
                temp_aa.append(aa_to_int[w])
            data_input.append(temp_aa)
    current, peak = tracemalloc.get_traced_memory()
    print(f"data_input end Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    return data_input, data_target
    
def create_model_and_train(best_model_path):
    global hist, my_callbacks, train_test_shuffle_count, continued
    model = model_FUTUSA(segmentation_size, aa_letter)
    model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['binary_accuracy','Recall'])
    if train_test_shuffle_count == 0:
        model.summary()
        if continued == True:
            print('Loading weights...')
            model.load_weights(best_model_path)
    else:
        print('Loading weights...')
        model.load_weights(best_model_path)
    hist = model.fit_generator(generator=training_generator, validation_data=validating_generator, epochs=20, callbacks=my_callbacks, class_weight={0: len(train_split[train_split['Target']==1]), 1: len(train_split[train_split['Target']==0])}, workers=4, use_multiprocessing=False, verbose=1)
    train_input.clear()
    train_target.clear()
    test_input.clear()
    test_target.clear()

my_callbacks = [CSVLogger(model_path + f'{target_go_num}_log.csv', separator=',', append=True),
EarlyStopping(patience=early_stopping_patience),
ModelCheckpoint(filepath=best_model_path, verbose=1, save_best_only=True, monitor='val_loss',mode='min')]

K.clear_session()

train_test_shuffle_count = 0
total_epoch = 0
tracemalloc.start()

if train_test_shuffle == True:

    while train_test_shuffle_count < maximum_train_test_shuffle:
        train_split, test_split = train_test_split(trainset, test_size=0.2, shuffle=True)
        train_split.reset_index(drop = True, inplace = True)
        test_split.reset_index(drop = True, inplace = True)
        print('Shuffled...')
        print(len(train_split[train_split['Target']==1]))
        print(len(test_split[test_split['Target']==1]))
        current, peak = tracemalloc.get_traced_memory()
        print(f"before Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        train_input, train_target = input_gen(train_split)
        test_input, test_target = input_gen(test_split)
        current, peak = tracemalloc.get_traced_memory()
        print(f"after input_gen Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        training_generator = DataGenerator(train_input,train_target,batch_size=256)
        validating_generator = DataGenerator(test_input,test_target,batch_size=256)
        current, peak = tracemalloc.get_traced_memory()
        print(f"after datagenerator Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        print(f'Train_Test shuffle count = {train_test_shuffle_count}')
        p = multiprocessing.Process(target=create_model_and_train(best_model_path=best_model_path))
        p.start()
        p.join()
        if train_test_shuffle_count == 0:
            val_best = min(hist.history['val_loss'])
        train_test_shuffle_count = train_test_shuffle_count+1
        print('val_best = ' + str(val_best) +', val_loss = ' + str(min(hist.history['val_loss'])))
        if val_best >= min(hist.history['val_loss']):
            patiented = 0
            print('patience initializing..')
            val_best = min(hist.history['val_loss'])
        else:
            patiented = patiented+1
            if patiented == train_test_shuffle_patience:
                print('training finished!')
                break
            else:
                print('patience +'+str(patiented))
else:
    p = multiprocessing.Process(target=create_model_and_train(best_model_path=False))
    p.start()
    p.join()
print('Done!')