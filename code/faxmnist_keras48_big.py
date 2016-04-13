import numpy as np
import struct
import cv2

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adadelta

np.random.seed(1024)

nb_dim = 48
nb_classes = 10
nb_epoch = 1000
batch_size = 16


def load_data(train=True):
    if train:
        fname_img = "../input/faxocr-training-{0}_train_images.idx3".format(nb_dim)
        fname_lbl = "../input/faxocr-training-{0}_train_labels.idx1".format(nb_dim)
    else:
        fname_img = "../input/faxocr-mustread-{0}_train_images.idx3".format(nb_dim)
        fname_lbl = "../input/faxocr-mustread-{0}_train_labels.idx1".format(nb_dim)

    with open(fname_img, "rb") as f:
        magic_nr, size, rows, cols = struct.unpack(">IIII", f.read(16))
        X = np.fromfile(f, dtype=np.uint8).astype(np.float32) / 255
        X = X.reshape(size, 1, rows, cols)
    with open(fname_lbl, "rb") as f:
        magic_nr, size = struct.unpack(">II", f.read(8))
        y = np_utils.to_categorical(np.fromfile(f, dtype=np.uint8))
    return X, y


def augment_data(X, y):
    X_list, y_list = [], []
    for i in xrange(X.shape[0]):
        img_i = X[i][0]
        X_list.append(img_i)
        y_list.append(y[i])
        for _ in range(10):
            angle = np.random.uniform(-30, 30)
            scale = np.random.uniform(0.9, 1.05)
            rotation_matrix = cv2.getRotationMatrix2D((nb_dim/2, nb_dim/2), angle, scale)
            img_rot = cv2.warpAffine(img_i, rotation_matrix, img_i.shape,
                                     flags=cv2.INTER_CUBIC)
            X_list.append(img_rot)
            y_list.append(y[i])
    return np.array(X_list).reshape(len(X_list), 1, nb_dim, nb_dim), np.array(y_list)

print "load data"
X_train, y_train = load_data()
X_test, y_test = load_data(train=False)

print "augment data"
X_train, y_train = augment_data(X_train, y_train)

print "build model"
model = Sequential()

# (1, 48, 48)
model.add(Convolution2D(64, 9, 9,
                        border_mode='valid',
                        input_shape=(1, nb_dim, nb_dim)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# (64, 20, 20)
model.add(ZeroPadding2D(padding=(4, 4)))
model.add(Convolution2D(64, 9, 9))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# (64, 10, 10)
model.add(Convolution2D(256, 5, 5))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# (256, 3, 3)
model.add(Convolution2D(256, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.4))

# (256, 1, 1)
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

adadelta = Adadelta(lr=0.1, rho=0.95, epsilon=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=adadelta)

print "fit"
weights_filepath = "../tmp/big_weights_{0}.hdf5".format(nb_dim)
earlystopping = EarlyStopping(monitor='val_loss', patience=10)
checkpointer = ModelCheckpoint(filepath=weights_filepath,
                               verbose=0, save_best_only=True)
model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
          verbose=1, validation_split=0.1, show_accuracy=True,
          callbacks=[earlystopping, checkpointer])
model.load_weights(weights_filepath)

print "best acc:",
print sum(model.predict_classes(X_test, verbose=0) == y_test.argmax(axis=1))*1.0 / X_test.shape[0]
