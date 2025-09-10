# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D


# def build_convlstm_model(input_shape, num_classes=3):
#     model = Sequential()
#     model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
#                          activation='relu',
#                          input_shape=input_shape,
#                          padding='same', return_sequences=True))
#     model.add(BatchNormalization())
#     model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
#                          activation='relu',
#                          padding='same', return_sequences=False))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=1, kernel_size=(3, 3),
#                      activation='sigmoid', padding='same'))

#     model.compile(optimizer='adam', loss='mse')
#     return model
    

# models/convLSTM.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D

def build_convlstm_model(input_shape):
    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape,
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3),
                         activation='relu',
                         padding='same', return_sequences=False))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=1, kernel_size=(3, 3),
                     activation='linear', padding='same'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

