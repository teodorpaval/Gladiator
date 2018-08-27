import numpy as np
import keras.backend as K
import keras_metrics
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import regularizers


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


class BinaryClassification:
    def __init__(self):
        self.model = Sequential()
        print "Creating Model..."
        self.create_baseline()
        print "Done."

    def create_baseline(self):
        self.model.add(Dense(29, input_dim=29, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                       activity_regularizer=regularizers.l1(0.01)))
        #self.model.add(Dense(29, input_dim=29, activation='relu'))
        self.model.add(Dense(15, activation='relu'))
        #self.model.add(Dense(22, activation='sigmoid'))
        self.model.add(Dropout(0.1))
        #self.model.add(Dense(33, activation='softmax'))
        self.model.add(Dense(15, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', recall])

    def train_model(self, x, y, epochs=5):
        print "Training the model..."

        self.model.fit(x, y, batch_size=10, epochs=epochs, verbose=1)

    def test_model(self, x, y):
        print("Testing the model...")
        score = self.model.evaluate(x, y, verbose=1, batch_size=32)
        print "recall: " + str(score[1]) + ", accuracy: " + str(score[2])
        filename = "models\model_"
        filename += str(score[1])
        filename += ".h5"

        print "Model Saved at: " + filename
        self.model.save(filename)
        return score
