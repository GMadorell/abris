from src import cochlea


def main():
    with open("data.txt", "r") as data_file, \
            open("config.json", "r") as config_file:
        data = cochlea.load(data_file, config_file)
    print data


if __name__ == "__main__":
    main()
