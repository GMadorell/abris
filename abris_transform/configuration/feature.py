from abris_transform.type_manipulation.translation.data_type_translation import type_name_to_data_type


class Feature(object):
    def __init__(self, name, characteristics_list):
        self.__name = str(name)
        self.__characteristics = characteristics_list
        self.__type_name = None

    def get_name(self):
        return self.__name

    def get_type_name(self):
        assert self.__type_name is not None
        return self.__type_name

    def set_type_name(self, type_name):
        self.__type_name = type_name

    def is_type_name(self, type_name):
        return type_name_to_data_type(self.get_type_name()) == type_name_to_data_type(type_name)

    def is_categorical(self):
        return self.has_characteristic("categorical")

    def is_target(self):
        return self.has_characteristic("target")

    def has_characteristic(self, characteristic):
        return characteristic.lower() in map(lambda string: string.lower(), self.__characteristics)

    def __repr__(self):
        return "Feature{Name=%s}" % self.get_name()
