from StringIO import StringIO
from src.cochlea import Cochlea


def main():
    with open("config.json", "r") as config_file:
        cochlea = Cochlea(config_file)

    with open("data.txt", "r") as data_file:
        data = cochlea.fit_transform(data_file)
    print data

    new_data = "10,Spain,1\n10,France,20"
    transformed_data = cochlea.transform(StringIO(new_data))
    print transformed_data


if __name__ == "__main__":
    main()
