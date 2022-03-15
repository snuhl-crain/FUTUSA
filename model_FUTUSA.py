
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, concatenate, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Model

def model_FUTUSA(seg_size, aa_letter):
    
    inputs = Input(shape=(seg_size,len(aa_letter)))
    
    conv1x1 = Conv1D(filters=21, kernel_size=1, strides=1, padding='same')(inputs)
    conv1x1_act = Activation('relu')(conv1x1)
    
    conv1d_2 = Conv1D(filters=32, kernel_size=2, strides=1, padding='same')(conv1x1_act)
    conv1d_2_nor = BatchNormalization()(conv1d_2)
    conv1d_2_act = Activation('relu')(conv1d_2_nor)
    maxpool1d_2 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d_2_act)
    
    conv1d_3 = Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(conv1x1_act)
    conv1d_3_nor = BatchNormalization()(conv1d_3)
    conv1d_3_act = Activation('relu')(conv1d_3_nor)
    maxpool1d_3 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d_3_act)
    
    conv1d_4 = Conv1D(filters=32, kernel_size=4, strides=1, padding='same')(conv1x1_act)
    conv1d_4_nor = BatchNormalization()(conv1d_4)
    conv1d_4_act = Activation('relu')(conv1d_4_nor)
    maxpool1d_4 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d_4_act)
    
    conv1d_7 = Conv1D(filters=32, kernel_size=7, strides=1, padding='same')(conv1x1_act)
    conv1d_7_nor = BatchNormalization()(conv1d_7)
    conv1d_7_act = Activation('relu')(conv1d_7_nor)
    maxpool1d_7 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d_7_act)
    
    conv1d_8 = Conv1D(filters=64, kernel_size=8, strides=1, padding='same')(conv1x1_act)
    conv1d_8_nor = BatchNormalization()(conv1d_8)
    conv1d_8_act = Activation('relu')(conv1d_8_nor)
    maxpool1d_8 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d_8_act)
    
    conv1d_13 = Conv1D(filters=64, kernel_size=13, strides=1, padding='same')(conv1x1_act)
    conv1d_13_nor = BatchNormalization()(conv1d_13)
    conv1d_13_act = Activation('relu')(conv1d_13_nor)
    maxpool1d_13 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d_13_act)
    
    conv1d_16 = Conv1D(filters=64, kernel_size=16, strides=1, padding='same')(conv1x1_act)
    conv1d_16_nor = BatchNormalization()(conv1d_16)
    conv1d_16_act = Activation('relu')(conv1d_16_nor)
    maxpool1d_16 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d_16_act)
    
    conv1d_31 = Conv1D(filters=64, kernel_size=31, strides=1, padding='same')(conv1x1_act)
    conv1d_31_nor = BatchNormalization()(conv1d_31)
    conv1d_31_act = Activation('relu')(conv1d_31_nor)
    maxpool1d_31 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d_31_act)
    
    conv1d_32 = Conv1D(filters=128, kernel_size=32, strides=1, padding='same')(conv1x1_act)
    conv1d_32_nor = BatchNormalization()(conv1d_32)
    conv1d_32_act = Activation('relu')(conv1d_32_nor)
    maxpool1d_32 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d_32_act)
    
    conv1d_61 = Conv1D(filters=128, kernel_size=61, strides=1, padding='same')(conv1x1_act)
    conv1d_61_nor = BatchNormalization()(conv1d_61)
    conv1d_61_act = Activation('relu')(conv1d_61_nor)
    maxpool1d_61 = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv1d_61_act)
    
    conc = concatenate([conv1x1_act,maxpool1d_2,maxpool1d_3,maxpool1d_4,maxpool1d_7,maxpool1d_8,maxpool1d_13,maxpool1d_16,maxpool1d_31,maxpool1d_32,maxpool1d_61], axis=2)
    flat = Flatten()(conc)
    
    drop_0 = Dropout(0.2)(flat)
    out_1 = Dense(128)(drop_0)
    out_1_nor = BatchNormalization()(out_1)
    out_1_act = Activation('relu')(out_1_nor)
    drop_1 = Dropout(0.2)(out_1_act)
    
    out_2 = Dense(96)(drop_1)
    out_2_nor = BatchNormalization()(out_2)
    out_2_act = Activation('relu')(out_2_nor)
    drop_2 = Dropout(0.2)(out_2_act)
    
    out_3 = Dense(64)(drop_2)
    out_3_nor = BatchNormalization()(out_3)
    out_3_act = Activation('relu')(out_3_nor)
    
    outputs = Dense(1, activation='sigmoid')(out_3_act)
    
    model = Model(inputs, outputs)
    
    return model