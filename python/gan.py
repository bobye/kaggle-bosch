from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.optimizers import SGD
from keras.initializations import normal
import numpy as np



batch_size = 128*128

numeric_feat=np.load('../data/numeric_feat.npz')['arr_0']
label=np.load('../data/label.npy')
data=numeric_feat[label==0,:]
print(data.shape)

data[data==-999]=0

print("Setting up decoder")
decoder = Sequential()
decoder.add(Dense(256, input_dim=data.shape[1], activation='relu', bias=False))
decoder.add(Dropout(0.5))
decoder.add(Dense(128, activation='relu'))
decoder.add(Dropout(0.5))
decoder.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, momentum=0.9)
decoder.compile(loss='binary_crossentropy', optimizer=sgd)

print("Setting up generator")
generator = Sequential()
generator.add(Dense(256, input_dim=128, activation='relu'))
generator.add(Dense(512, activation='relu'))
generator.add(Dense(data.shape[1], activation='linear'))

generator.compile(loss='mean_squared_error', optimizer=sgd)

print("Setting up combined net")
gen_dec = Sequential()
gen_dec.add(generator)
decoder.trainable=False
gen_dec.add(decoder)


gen_dec.compile(loss='binary_crossentropy', optimizer=sgd)




y_decode = np.ones(2*batch_size)
y_decode[:batch_size] = 0.
y_gen_dec = np.ones(batch_size)


re=0.5
rd=0.5
for i in range(1000):
    zmb = np.random.uniform(-1, 1, size=(batch_size, 128)).astype('float32')
    xmb = data[np.random.randint(0,data.shape[0],batch_size),:]
    if (re > rd or i % 5 == 0) and i % 7 != 0:
        re = gen_dec.train_on_batch(zmb,y_gen_dec)
        if i % 10 == 0:
            print('E:' + str(np.exp(-re)) + ' D:' + str(np.exp(-rd)))
    else:
        rd = decoder.train_on_batch(np.vstack([generator.predict(zmb),xmb]),y_decode)
        if i % 10 == 0:
            print('E:' + str(np.exp(-re)) + ' D:' + str(np.exp(-rd)))
        

hidden = Sequential()
hidden.add(Dense(256, weights=decoder.layers[0].get_weights(), input_dim=data.shape[1], activation='relu', bias=False))
hidden.add(Dense(128, weights.decoder.layers[2].get_weights(), activation='linear'))
numeric_hidden=hidden._predict(numeric_feat)
np.savez_compressed('numeric_hidden.npz', numeric_hidden)
