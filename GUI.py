import tkMessageBox
from Tkinter import *
from tkMessageBox import *
from datareader import DatasetReader
from binary_classification import BinaryClassification


class AppData:
    def __init__(self, master):
        self.button_load_data = Button(master, text="Load Dataset", command=self.load_data)
        self.button_load_model = Button(master, text="Load Model", command=self.new_model, state=DISABLED)

        self.string_train_data = StringVar()
        self.string_train_data.set("Clean:\nFraud:\nTotal:")
        self.display_train_data = Label(master, textvariable=self.string_train_data, anchor="w", width=25, bg="white")
        self.string_test_data = StringVar()
        self.string_test_data.set("Clean:\nFraud:\nTotal:")
        self.display_test_data = Label(master, textvariable=self.string_test_data, anchor="w", width=25, bg="white")

        self.string_train_log = StringVar()
        self.string_test_log = StringVar()
        self.display_test_log = Label(master, textvariable=self.string_test_log, anchor="w", height=10, width=25, bg="white")
        self.display_train_log = Label(master, textvariable=self.string_train_log, anchor="w", height=10, width=25, bg="white")

        self.label_test_data = Label(master, text="Testing Data")
        self.label_train_data = Label(master, text="Training Data")

        self.button_train = Button(master, text="Train Model", command=self.train, state=DISABLED)
        self.button_test = Button(master, text="Test Model", command=self.test, state=DISABLED)

        self.grid_layout()

        # self.button_load_data.pack()
        # self.button_load_model.pack()
        #
        # self.button_train.pack()
        # self.button_test.pack()

    def grid_layout(self):
        self.button_load_data.grid(row=0, column=0, padx=(15, 15), pady=(10, 10), sticky=N)
        self.display_train_data.grid(row=0, column=1, pady=(10, 2))
        self.display_test_data.grid(row=0, column=2, pady=(10, 2), padx=(10, 15))
        self.label_train_data.grid(row=1, column=1)
        self.label_test_data.grid(row=1, column=2)
        self.button_load_model.grid(row=2, column=0, padx=(15, 15), pady=(10, 10), sticky=N)
        self.display_train_log.grid(row=2, column=1, pady=(10, 5))
        self.display_test_log.grid(row=2, column=2, pady=(10, 5), padx=(10, 10))
        self.button_train.grid(row=3, column=1)
        self.button_test.grid(row=3, column=2)

    def count_data(self, dataset=[]):
        fraud_count = len([fraud for fraud in dataset if fraud == 1])
        clean_count = len(dataset) - fraud_count
        return "Clean: %s\nFraud: %s\nTotal: %s" % (str(clean_count), str(fraud_count), str(len(dataset)))

    def load_data(self):
        #print(self.undersampling.get())
        dr = DatasetReader("creditcard.csv")
        dataset = dr.get_dataset()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dr.get_data()
        self.button_load_model['state'] = NORMAL
        print(self.y_train[:5])
        self.string_train_data.set(self.count_data(self.y_train))
        self.string_test_data.set(self.count_data(self.y_test))
        tkMessageBox.showinfo("Success!", "The dataset has been loaded successfully.")

    def new_model(self):
        self.model = BinaryClassification()
        tkMessageBox.showinfo("Success!", "The model has been created successfully")
        self.button_train['state'] = NORMAL

    def train(self):
        self.model.train_model(self.x_train, self.y_train, 100)
        self.button_test['state'] = NORMAL

    def test(self):
        score = self.model.test_model(self.x_test, self.y_test)
        #print "loss: " + str(score[0]) + "; accuracy: " + str(score[1]) + "; recall: " + str(score[2])
        tkMessageBox.showinfo("Results", "Evaluation finished!\nRecall: " + str(score[1]))


root = Tk()

root.title("Gladiator")
app_data = AppData(root)

root.mainloop()