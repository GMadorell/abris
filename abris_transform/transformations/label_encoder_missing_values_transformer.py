from sklearn.preprocessing import LabelEncoder
import numpy as np

from abris_transform.transformations.base_transformer import BaseTransformer


class LabelEncoderMissingValuesTransformer(BaseTransformer):
    def __init__(self):
        self.__label_encoder = LabelEncoder()

    def fit(self, data):
        self.__label_encoder.fit(data)

    def transform(self, data):
        classes = self.__label_encoder.classes_

        if not self.__column_contains_all(data, classes):
            if "<Unknown>" not in classes:
                self.__label_encoder.classes_ = np.append(classes, "<Unknown>")
            data = data.map(lambda s: '<Unknown>' if s not in self.__label_encoder.classes_ else s)

        return self.__label_encoder.transform(data)

    def __column_contains_all(self, data, classes):
        return data.str.contains("|".join(classes)).all()
