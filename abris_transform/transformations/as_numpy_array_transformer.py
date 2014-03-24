from abris_transform.transformations.base_transformer import BaseTransformer
import numpy as np


class AsNumpyArrayTransformer(BaseTransformer):
    """
    A transformer that does the bare minimum to not break anything.
    Used whenever the data should not be modified but be returned as
    a numpy array compatible datastructure.
    """

    def fit(self, data):
        pass

    def transform(self, data):
        return np.asarray(data)