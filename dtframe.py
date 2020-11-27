import pandas as pd
import math


def calculate_entropy(event_probs):
    entropy = 0
    for event_prob in event_probs:
        try:
            entropy += event_prob * math.log2(event_prob)
        except ValueError:
            pass
    return -entropy if entropy != 0 else 0


class DTFrame(pd.DataFrame):
    def info_gain(self, attribute):
        info = 0
        res_vals = list(self[self.columns[-1]])
        attr_vals = list(self[attribute])
        uniq_attr_vals = set(attr_vals)
        uniq_res_vals = set(res_vals)
        for uniq_attr_val in uniq_attr_vals:
            attr_val_assoc_results = [res_vals[i] for i,x in enumerate(attr_vals) if x == uniq_attr_val]
            args = []
            for uniq_res_val in uniq_res_vals:
                args.append(attr_val_assoc_results.count(uniq_res_val) / len(attr_val_assoc_results))
            entropy = calculate_entropy(args)
            info += (attr_vals.count(uniq_attr_val) / len(attr_vals)) * entropy
        return self.current_entropy()-info

    def current_entropy(self):
        results = list(self[self.columns[-1]])
        events = set(results)
        event_probs = []
        for event in events:
            event_probs.append(results.count(event) / len(results))
        return calculate_entropy(event_probs)

    def next_best_attribute_for_info_gain(self):
        lst = []
        for column in self.columns[:-1]:
            lst.append((self.info_gain(column), column))
        lst.sort(reverse=True)
        return lst[0]

    def decision_tree(self):
        print()
        _, attribute = self.next_best_attribute_for_info_gain()
        queue = [(self, attribute)]

        while queue:
            df, attribute = queue.pop(0)
            print(attribute)

            for uniq_attr_val in set(df[attribute]):
                df_attr = df.split_on_attribute(attribute, uniq_attr_val)
                if df_attr.current_entropy():
                    _, new_attribute = df_attr.next_best_attribute_for_info_gain()
                    print(f'\t{uniq_attr_val} - {new_attribute}')
                    queue.append((df_attr, new_attribute))
                else:
                    if len(df_attr[df_attr.columns[-1]]) > 0:
                        val = list(df_attr[df_attr.columns[-1]])[0]
                        print(f'\t{uniq_attr_val} - {val}')

    def split_on_attribute(self, attribute, attr_val):
        return DTFrame(self[self[attribute] == attr_val])

    @classmethod
    def read_txt(cls, filename):
        file = open(filename, 'r')
        lines = file.readlines()
        dic = {}
        for topic in lines[0].split():
            dic[topic] = []

        for i in range(2, 102):
            line = lines[i].split()
            for j, topic in enumerate(dic):
                dic[topic].append(line[j])

        return cls(dic)






