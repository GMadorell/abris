from abris_transform.transformations.base_transformer import BaseTransformer


class NullTransformer(BaseTransformer):
    """
    A transformer that does nothing. Can be used whenever a transformer
    is needed but no data transformation needs to happen.
    """

    def fit(self, data):
        pass

    def transform(self, data):
        return data