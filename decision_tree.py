import pandas as pd

class Question:
    def __init__(self, topic):
        self.topic = topic
        self.options = []
        self.results = []

    def new_case(self, option, result):
        self.options.append(option)
        self.results.append(result)

    def get_result(self, choice):
        for i, option in enumerate(self.options):
            if option == choice:
                return self.results[i]
        raise IndexError(f"No result for choice {choice}")

    def __repr__(self):
        string = ''
        string += str(self.topic) + '\n'
        for i in range(len(self.options)):
            string += f'\t{self.options[i]} - {self.results[i]}\n'
        return string


class DecisionTree:
    def __init__(self):
        self.initial_question = None
        self.questions = {}
        self.in_order = []

    def process(self, df, i):
        entry = df.iloc[i]
        assert isinstance(entry, pd.Series), "entry must be a pandas series"
        assert len(self.questions) > 0, "questions cannot be empty"
        first_q = self.questions[self.initial_question]
        next_col = first_q.get_result(entry[first_q.topic])
        while next_col in df.columns:
            q = self.questions[next_col]
            next_col = q.get_result(entry[q.topic])
        return next_col

    def add(self, question):
        assert isinstance(question, Question), "Decision Tree add method only accepts Question instances"
        if len(self.questions) == 0:
            self.initial_question = question.topic
        self.questions[question.topic] = question
        self.in_order.append(question)

    def __repr__(self):
        string = ''
        for q in self.in_order:
            string += repr(q)
        return string
