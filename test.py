import pytest
from main import *
from dtframe import DTFrame


@pytest.fixture
def wait_for_table():
    dic = {
        'Alt': [True, True, False, True, True, False, False, False, False, True, False, True],
        'Bar': [False, False, True, False, False, True, True, False, True, True, False, True],
        'Fri': [False, False, False, True, True, False, False, False, True, True, False, True],
        'Hun': [True, True, False, True, False, True, False, True, False, True, False, True],
        'Pat': ['Some', 'Full', 'Some', 'Full', 'Full', 'Some', 'None', 'Some', 'Full', 'Full', 'None', 'Full'],
        'Price': [3, 1, 1, 1, 3, 2, 1, 2, 1, 3, 1, 1],
        'Rain': [False, False, False, False, False, True, True, True, True, False, False, False],
        'Res': [True, False, False, False, True, True, False, True, False, True, False, False],
        'Type': ['French', 'Thai', 'Burger', 'Thai', 'French', 'Italian', 'Burger', 'Thai', 'Burger', 'Italian', 'Thai',
                 'Burger'],
        'Est': [1, 3, 1, 2, 4, 1, 1, 1, 4, 2, 1, 3],
        'Target Wait': [True, False, True, True, False, True, False, True, False, False, False, True]
    }
    return DTFrame(dic)


@pytest.fixture
def play_golf():
    dic = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy',
                    'Sunny', 'Overcast', 'Overcast', 'Rainy'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cold', 'Cold', 'Cold', 'Mild', 'Cold', 'Mild', 'Mild', 'Mild',
                        'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal',
                     'High', 'Normal', 'High'],
        'Wind': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'False', 'False', 'True', 'True',
                 'False', 'True'],
        'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    return DTFrame(dic)


@pytest.fixture
def train_house_votes():
    return DTFrame.read_txt('train-house-votes-1984.txt')


@pytest.fixture
def test_house_votes():
    return DTFrame.read_txt('test-house-votes-1984.txt')


def test_entropy(wait_for_table):
    print()
    print(wait_for_table)
    with pytest.raises(KeyError):
        wait_for_table.info_gain('Pats')
    assert round(wait_for_table.info_gain('Pat'), 3) == 0.541
    assert round(wait_for_table.info_gain('Type'), 3) == 0


def test_next_attr_split(wait_for_table):
    assert wait_for_table.next_best_attribute_for_info_gain()[1] == 'Pat'


def test_set_entropy(wait_for_table):
    assert wait_for_table.current_entropy() == 1


def test_decision_tree(wait_for_table, play_golf):
    print(wait_for_table.decision_tree())
    print(play_golf.decision_tree())


def test_house_votes_tree(train_house_votes, test_house_votes):
    print(train_house_votes.decision_tree())
    print(test_house_votes.decision_tree())


def test_accuracy(train_house_votes, test_house_votes):
    dt_train = train_house_votes.decision_tree()
    dt_test = test_house_votes.decision_tree()

    print('----Train House Votes----')
    print(dt_train)
    print('Tested with test house votes:')
    test_house_votes.test_decision_tree(dt_train)

    print('----Test House Votes----')
    print(dt_test)
    print('Tested with train house votes:')
    train_house_votes.test_decision_tree(dt_test)


