import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path

_DIR = Path(__file__).parent


class chatModel:
    def __init__(self):
        self.model = tf.keras.models.load_model(str(_DIR / 'chatbot_model1.h5'))
        df = pd.read_csv(str(_DIR / 'data_tutorial.csv'))
        self.symp = list(df.iloc[:, :26])
        self.ms = list(df.iloc[:, -9:])

    def msSuggest(self, inputList):
        df = pd.DataFrame([inputList], columns=self.symp)
        prediction = self.model.predict(df)
        ms_index = np.where(prediction[0] == max(prediction[0]))[0][0]
        return self.ms[ms_index] + f' ({max(prediction[0]):.1%})'
