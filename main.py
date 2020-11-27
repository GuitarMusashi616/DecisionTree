from dtframe import *
from test import *

if __name__ == '__main__':
    df_1 = DTFrame.read_txt('train-house-votes-1984.txt')
    print("\n\n----TRAIN HOUSE VOTES----")
    df_1.decision_tree()

    df_2 = DTFrame.read_txt('test-house-votes-1984.txt')
    print('\n\n----TEST HOUSE VOTES----')
    df_2.decision_tree()

