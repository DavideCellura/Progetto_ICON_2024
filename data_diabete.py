from seaborn.matrix import heatmap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DIABETES_DATA_FILE = "diabetes.csv"
TARGET = "diabetes"


class diabetes_data:

    def __init__(self):
        self.data = pd.read_csv(DIABETES_DATA_FILE).dropna()
        del self.data['gender']
        del self.data['smoking_history']
        self.features_list = list(self.data.columns)

    def get_data(self):
        return self.data


    def plot_diabetes(self):
        plt.style.use("ggplot")
        self.data["diabetes"].value_counts().plot.bar(
            title='diabetes', rot=0)
        plt.show()

    def plot_ages(self):
        plt.style.use("ggplot")
        list_ages = []

        for a in self.data["age"]:
            for e in range(0, 100, 10):
                if a >= e and a <= (e+9):
                    label = "%d-%d" % (e, (e+9))
                    list_ages.append(label)
        pd.DataFrame(list_ages).value_counts().plot.bar(
            title='Ages', rot=0)
        plt.show()

    def get_training_data(self):

        y = self.data[[TARGET]].values
        x = self.data.drop(TARGET, axis='columns').values

        return x, y
    
    def get_medium_values_diabetes(self):

        medium_values = {}
        positives = self.data[self.data['diabetes'] == 1]
             
        medium_values['bmi'] = positives['bmi'].mean()

        return medium_values