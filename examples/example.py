from StringIO import StringIO
from pprint import pprint

from abris_transform.abris import Abris


def main():
    with open("config.json", "r") as config_file:
        abris = Abris(config_file)

    initial_data = "10,Spain,3.5,True,100\n" \
                   "12,France,,True,20\n" \
                   "14,Germany,10.5,False,50\n" \
                   "12,France,2.5,True,20\n" \
                   "12,,2.5,True,20\n" \
                   "12,  France   ,2.5,True,20\n" \
                   "12,France,2.5,True,20\n" \
                   "12,France,2.5,True,20\n" \
                   "12,France,2.5,True,20\n" \
                   "12,France,2.5,True,20\n"

    train, test = abris.prepare(StringIO(initial_data))
    pprint(train)
    print "---"
    pprint(test)
    print "---"

    new_data = "10,Spain,1,True\n" \
               "10,France,20,False"
    transformed_data = abris.apply(StringIO(new_data))
    pprint(transformed_data)


if __name__ == "__main__":
    main()
