from datareader import DatasetReader
from binary_classification import BinaryClassification


dr = DatasetReader("creditcard.csv")

dataset = dr.get_dataset()
(x_train, y_train), (x_test, y_test) = dr.get_data()

#print len(dataset)
model = BinaryClassification()
model.train_model(x_train, y_train, 20)
model.test_model(x_test, y_test)

