from keras.models import Model
from keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Activation
from keras.layers.merge import concatenate
from keras import backend as K

# Design our model architecture
def create_model(img_hw, input_channels, dropout_value):
    n_ch_exps = [4, 5, 6, 7, 8, 9]
    k_size = (3, 3)                  #size of filter kernel
    k_init = 'he_normal'             #kernel initializer

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (input_channels, img_hw, img_hw)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_hw, img_hw, input_channels)


    inp = Input(shape=input_shape)
    lambd = Lambda(lambda x: x / 255) (inp)

    encodeds = []
    # encoder
    enc = lambd
    #print(n_ch_exps)
    for l_idx, n_ch in enumerate(n_ch_exps):
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Dropout(dropout_value)(enc)
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        encodeds.append(enc)
        #print(l_idx, enc)
        if n_ch < n_ch_exps[-1]:  #do not run max pooling on the last encoding/downsampling step
            enc = MaxPooling2D(pool_size=(2,2))(enc)
    
    # decoder
    dec = enc
    #print(n_ch_exps[::-1][1:])
    decoder_n_chs = n_ch_exps[::-1][1:]
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Dropout(dropout_value)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)

    outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(dec)

    model = Model(inputs=[inp], outputs=[outp])
    
    return model

# Design our model architecture
def create_model_ff(img_hw, input_channels, dropout_value):
    #n_ch_exps = [4, 5, 6, 7, 8, 9]
    n_ch_exps = [6,7,8,9,10]
    #n_ch_exps = [6,7,8,9,10]
    #n_ch_exps = [6,7,8]
    k_size = (3, 3)                  #size of filter kernel
    k_init = 'he_normal'             #kernel initializer

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (input_channels, img_hw, img_hw)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_hw, img_hw, input_channels)


    inp = Input(shape=input_shape)
    #lambd = Lambda(lambda x: x / 255) (inp)

    encodeds = []
    # encoder
    #enc = lambd
    enc = inp
    #print(n_ch_exps)
    for l_idx, n_ch in enumerate(n_ch_exps):
        #enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', use_bias=False, kernel_initializer=k_init)(enc)
        enc = BatchNormalization()(enc)
        enc = Activation('relu')(enc)

        enc = Dropout(dropout_value)(enc)
        #enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', use_bias=False, kernel_initializer=k_init)(enc)
        enc = BatchNormalization()(enc)
        enc = Activation('relu')(enc)

        encodeds.append(enc)
        #print(l_idx, enc)
        if n_ch < n_ch_exps[-1]:  #do not run max pooling on the last encoding/downsampling step
            enc = MaxPooling2D(pool_size=(2,2))(enc)
    
    # decoder
    dec = enc
    #print(n_ch_exps[::-1][1:])
    decoder_n_chs = n_ch_exps[::-1][1:]
    #print(decoder_n_chs)
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        #dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), padding='same', use_bias=False, kernel_initializer=k_init)(dec)
        dec = BatchNormalization()(dec)
        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
        #dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', use_bias=False, kernel_initializer=k_init)(dec)
        dec = BatchNormalization()(dec)
        dec = Activation('relu')(dec)

        dec = Dropout(dropout_value)(dec)
        #dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', use_bias=False, kernel_initializer=k_init)(dec)
        dec = BatchNormalization()(dec)
        dec = Activation('relu')(dec)

    outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(dec)

    model = Model(inputs=[inp], outputs=[outp])
    
    return model


# Design our model architecture
def create_model_rocks(img_hw, input_channels, dropout_value):
    #n_ch_exps = [4, 5, 6, 7, 8, 9]
    #n_ch_exps = [4,5,6,7,8]
    n_ch_exps = [4,5,6,7]
    #n_ch_exps = [6,7,8,9,10]
    #n_ch_exps = [6,7,8]
    k_size = (3, 3)                  #size of filter kernel
    k_init = 'he_normal'             #kernel initializer

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (input_channels, img_hw, img_hw)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_hw, img_hw, input_channels)


    inp = Input(shape=input_shape)
    lambd = Lambda(lambda x: x / 255) (inp)

    encodeds = []
    # encoder
    enc = lambd
    #print(n_ch_exps)
    for l_idx, n_ch in enumerate(n_ch_exps):
        #enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', kernel_initializer=k_init)(enc)
        end = BatchNormalization()(enc)
        end = Activation('relu')(enc)

        enc = Dropout(dropout_value)(enc)
        #enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', kernel_initializer=k_init)(enc)
        end = BatchNormalization()(enc)
        end = Activation('relu')(enc)

        encodeds.append(enc)
        #print(l_idx, enc)
        if n_ch < n_ch_exps[-1]:  #do not run max pooling on the last encoding/downsampling step
            enc = MaxPooling2D(pool_size=(2,2))(enc)
    
    # decoder
    dec = enc
    #print(n_ch_exps[::-1][1:])
    decoder_n_chs = n_ch_exps[::-1][1:]
    #print(decoder_n_chs)
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
        #dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', kernel_initializer=k_init)(dec)
        dec = BatchNormalization()(dec)
        dec = Activation('relu')(dec)

        dec = Dropout(dropout_value)(dec)
        #dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', kernel_initializer=k_init)(dec)
        dec = BatchNormalization()(dec)
        dec = Activation('relu')(dec)

    outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(dec)

    model = Model(inputs=[inp], outputs=[outp])
    
    return model

#model = create_model(img_hw=128, 3)
#model.summary()
