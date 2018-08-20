import keras
from keras.layers import *
from keras.applications import *
from keras.models import *
import matplotlib.pyplot as plt

def vgg_19_lstm(input_shape=(224, 224, 3), timesteps=5, dropout_value=0.5, num_class=2, lstm_size=1024):
    _new_input_shape = tuple([timesteps]) + input_shape
    _input_sequences = Input(shape=_new_input_shape)
    _vgg_19_model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    _cnn_feature_maps  = TimeDistributed(_vgg_19_model,input_shape=_new_input_shape)(_input_sequences)
    #_cnn_feature_maps  = TimeDistributed(Conv2D(filters=128,kernel_size=(3,3)))(_cnn_feature_maps)
    _cnn_feature_maps  = TimeDistributed(Flatten())(_cnn_feature_maps)
    _sequence_features = TimeDistributed(Dense(units=lstm_size,activation='relu'))(_cnn_feature_maps)
    _global_features   = Lambda(lambda x:x[:,0,:])(_sequence_features)
    _lstm_features = LSTM(units=lstm_size, activation='tanh', return_sequences=True, dropout=dropout_value)(_sequence_features)
    _lstm_features = LSTM(units=lstm_size, activation='tanh', return_sequences=False, dropout=dropout_value)(_lstm_features)
    _global_features = Reshape(target_shape=(lstm_size,))(_global_features)
    _global_features = Dense(units=lstm_size, activation='relu')(_global_features)
    _combined_features = Concatenate()([_global_features,_lstm_features])
    #_combined_features = Multiply()([_global_features,_lstm_features])
    _combined_features = Dropout(rate=dropout_value)(_combined_features)
    _combined_features = Dense(units=lstm_size, activation='relu')(_combined_features)
    _combined_features = Dropout(rate=dropout_value)(_combined_features)
    _predictions = Dense(units=num_class, activation='softmax')(_combined_features)
    _model = Model(inputs=[_input_sequences],outputs=[_predictions])
    #for layer in _model.layers[:2]:
    #    layer.trainable = False #Freezee VGG weights
    _optimizer = keras.optimizers.Adam(lr=0.00001)
    _model.compile(optimizer=_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return _model

def vgg_19_lstm_v2(input_shape=(224, 224, 3), timesteps=5, dropout_value=0.5, num_class=2, lstm_size=1024):
    _new_input_shape = tuple([timesteps]) + input_shape
    _input_sequences = Input(shape=_new_input_shape)
    _vgg_19_model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    _cnn_feature_maps  = TimeDistributed(_vgg_19_model,input_shape=_new_input_shape)(_input_sequences)
    _cnn_feature_maps  = TimeDistributed(Flatten())(_cnn_feature_maps)
    _sequence_features = TimeDistributed(Dense(units=lstm_size,activation='relu'))(_cnn_feature_maps)
    _global_features   = Lambda(lambda x:x[:,0,:])(_sequence_features)
    _lstm_features = LSTM(units=lstm_size, activation='tanh', return_sequences=False, dropout=dropout_value)(_sequence_features)
    _global_features = Reshape(target_shape=(lstm_size,))(_global_features)
    _global_features = Dense(units=lstm_size, activation='relu')(_global_features)
    _combined_features = Concatenate()([_global_features,_lstm_features])
    _combined_features = Dropout(rate=dropout_value)(_combined_features)
    _combined_features = Dense(units=lstm_size, activation='relu')(_combined_features)
    _combined_features = Dropout(rate=dropout_value)(_combined_features)
    _predictions = Dense(units=num_class, activation='softmax')(_combined_features)
    _model = Model(inputs=[_input_sequences],outputs=[_predictions])
    _optimizer = keras.optimizers.Adam(lr=0.00001)
    _model.compile(optimizer=_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return _model

class save_model_during_train(keras.callbacks.Callback):
    def __init__(self,i_save_model_path = None):
        self.i = 0
        self.save_model_path = i_save_model_path
    def on_epoch_end(self, epoch, logs=None):
        self.i +=1
        if self.save_model_path is not None:
            self.model.save(filepath='{}/model-epoches-{}.h5'.format(self.save_model_path,self.i))

class draw_loss_curve(keras.callbacks.Callback):
    def __init__(self,i_save_model_path=None):
        self.i = 0
        self.x = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.save_model_path = i_save_model_path
    def on_train_begin(self, logs=None):
        self.i = 0
        self.x =[0]
        self.train_loss = []
        self.val_loss = []
        self.train_acc = [0]
        self.val_acc = [0]
    def on_epoch_end(self, epoch, logs=None):
        self.i += 1
        if self.i == 1:
            self.train_loss.append(logs.get('loss'))
            self.val_loss.append(logs.get('val_loss'))
        self.x.append(self.i)
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        optimizer = self.model.optimizer
        current_lr = keras.backend.eval(optimizer.lr)
        print('Current learning rate = {:10f}...'.format(current_lr))
        print(self.params)
        if self.save_model_path is not None:
            self.model.save(filepath='{}/model-epoches-{}.h5'.format(self.save_model_path,self.i))
    def on_train_end(self, logs=None):
        self.x.append(self.i)
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.grid(color = 'k',linestyle = 'dashdot', linewidth = 0.25)
        ax1.plot(self.x, self.train_loss, label='loss')
        ax1.plot(self.x, self.val_loss, label='val_loss')
        ax1.legend(['loss', 'val_loss'])
        #ax1.legend(['loss','val_loss'],loc='upper center')
        ax2.grid(color='k', linestyle='dashdot', linewidth=0.25)
        ax2.plot(self.x, self.train_acc, label='acc')
        ax2.plot(self.x, self.val_acc, label='val_acc')
        ax2.legend(['acc', 'val_acc'])
        #ax2.legend(['acc','val_acc'],loc='upper center')
        plt.show()

def lrate_schedule(epoch, c_lrate):
    drop_period = 2
    drop_factor = 0.1
    if epoch==0:
        return c_lrate
    elif epoch%drop_period==0:
        return c_lrate*drop_factor
    else:
        return c_lrate
#-------------------------
if __name__=="__main__":
    model = vgg_19_lstm()
    print(model.summary())
