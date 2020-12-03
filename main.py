from dtframe import *
from test import *


if __name__ == '__main__':
    df_1 = DTFrame.read_txt('train-house-votes-1984.txt')
    df_2 = DTFrame.read_txt('test-house-votes-1984.txt')

    print("\n\n----Train House Votes----")
    dt_1 = df_1.decision_tree()
    print(dt_1)
    print("Tested with test house votes:")
    df_2.test_decision_tree(dt_1)

