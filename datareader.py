import numpy as np
import random
import csv
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


class DatasetReader:
    def __init__(self, filename, undersampled=True):
        self._filename = filename
        self._dataset_large = np.array([])
        self._dataset = np.array([])
        self._x_train = []
        self._y_train = []
        self._x_test = []
        self._y_test = []
        print(undersampled)
        print "Reading input file..."
        self.read_csv(undersampled)
        print "Done."

    def get_data(self):
        return (self._x_train, self._y_train), (self._x_test, self._y_test)

    def get_dataset(self):
        return self._dataset

    def standardize(self, a):
        a_stand = (a - a.mean() * np.ones(len(a))) / (a.std())
        return a_stand

    def process_data(self, xy=[], xy_undersample=[]):
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
            np.array([x[:-1] for x in xy_undersample]), [y[-1] for y in xy_undersample], test_size=0.33
        )
        # self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
        #    np.array([x[:-1] for x in xy]), [y[-1] for y in xy], test_size=0.33
        # )
        # self._x_test = np.array([x[:-1] for x in xy])
        # self._y_test = [y[-1] for y in xy]


    def undersample_data(self, a):
        fraud_data = [fraud for fraud in a if fraud[-1] == 1]
        clean_data = random.sample([clean for clean in a if clean[-1] == 0], len(fraud_data))
        return fraud_data + clean_data

    def read_csv(self, undersampled=True):
        # reading the whole dataset
        with open(self._filename) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            next(reader, None)# skip header
            raw_data = [np.array(a[1:], dtype=float) for a in reader]

        self._dataset = np.array(raw_data)
        self._dataset = normalize(self._dataset, axis=0, norm="max")
        np.random.shuffle(self._dataset)
        self.process_data(self._dataset, self.undersample_data(raw_data))
