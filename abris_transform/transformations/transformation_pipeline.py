from abris_transform.transformations.base_transformer import BaseTransformer


class TransformationPipeline(BaseTransformer):
    def __init__(self):
        self.__transformers = []

    def add_transformer(self, transformer):
        self.__transformers.append(transformer)

    def fit(self, data):
        for transformer in self.__transformers:
            transformer.fit(data)

    def transform(self, data):
        for transformer in self.__transformers:
            data = transformer.transform(data)
        return data
