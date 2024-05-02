import tensorflow as tf
import pandas as pd
import numpy as np


class chatModel:
    def __init__(self):
        self.model = tf.keras.models.load_model('chatbot_model1.h5')
        self.symp = list(pd.read_csv('data_tutorial.csv').iloc[:, :26])
        self.ms = list(pd.read_csv('data_tutorial.csv').iloc[:, -9:])

    def msSuggest(self, inputList):
        df = pd.DataFrame([inputList], columns=self.symp)
        prediction = self.model.predict(df)
        ms_index = np.where(prediction[0] == max(prediction[0]))[0][0]
        return self.ms[ms_index] + ' (' + str(max(prediction[0])) + ')'
