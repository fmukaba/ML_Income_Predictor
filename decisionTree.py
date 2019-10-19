
import pandas as pd
import math
from graphviz import Digraph


class ID3:

    @staticmethod
    # calculate the entropy of a dataframe
    def __entropy(df: pd.DataFrame):
        df_label = df.columns[-1]
        values = df[df_label].value_counts()
        total_count = len(df)
        entropy = 0
        for val in values:
            entropy += (val/total_count) * math.log((val/total_count), 2)
        return entropy * -1

    @staticmethod
    # calculate information gain of a feature
    def __info_gain(df: pd.DataFrame, attribute: str):
        entropy_before = ID3.__entropy(df)
        entropy_after = 0
        for value in df[attribute].unique():
            partition = df[df[attribute] == value]
            entropy_after += len(partition) / len(df) * ID3.__entropy(partition)
        return entropy_before-entropy_after

    @staticmethod
    # return the best feature to partition the dataframe by
    def __best_feature(df: pd.DataFrame):
        best_feature = ""
        best_gain = 0
        features_list = list(df.columns.values[:-1])
        for feature in features_list:
            gain = ID3.__info_gain(df, feature)
            if gain > best_gain:
                best_feature = feature
                best_gain = gain
        return best_feature

    @staticmethod
    # given a list of features, return the best feature to partition the dataframe by
    def __best_feature_list(df: pd.DataFrame, features_list: list):
        best_feature = features_list[0]
        best_gain = ID3.__info_gain(df, best_feature)
        for features in features_list:
            gain = ID3.__info_gain(df, features)
            if gain > best_gain:
                best_feature = features
                best_gain = gain
        return best_feature

    @staticmethod
    # generate a decision tree
    def build_tree(df: pd.DataFrame) -> {}:
        if len(df) == 0:
            return {}
        features_list = list(df.columns.values[:-1])
        return ID3.__build_tree(df, features_list)

    @staticmethod
    # recursive helper to build the tree
    def __build_tree(df: pd.DataFrame, features_list: list):
        label = df.columns[-1]

        # if there is no more attribute to partition
        if len(features_list) == 0:
            # return most common class label
            return df[label].value_counts().idxmax()

        # if the entropy is zero
        if ID3.__entropy(df) == 0:
            # return the first instance of the class label
            return df[label].iloc[0]
        # get best feature
        best_feature = ID3.__best_feature_list(df, features_list)
        features_copy = features_list.copy()
        features_copy.remove(best_feature)
        # get all possible values for best feature
        values_bf = df[best_feature].unique()
        root_tree, attr_tree = {}, {}
        # partition dataframe
        for value in values_bf:
            partition = df[df[best_feature] == value]
            if len(partition) == 0:
                attr_tree[value] = df[label].value_counts().idxmax()
            else:
                attr_tree[value] = ID3.__build_tree(partition, features_copy)
        # put partitioned frame under best feature's dictionnary
        root_tree[best_feature] = attr_tree
        return root_tree


def classify_row(row :pd.Series, decision_tree: {}) -> bool:
    # while dealing with a dictionary and not a class label (string)
    while isinstance(decision_tree, dict):
        # iterate through the row and get the value of the feature
        feature = next(iter(decision_tree))
        value = row[feature]
        if value not in decision_tree[feature]:
            return None
        # break dow into following the path on the tree
        decision_tree = decision_tree[feature][value]
    # compare decision tree value (string) to target of the row
    if decision_tree == row[-1]:
        return True
    else:
        return False

def calculate_accuracy(testing_df: pd.DataFrame, decision_tree: {}):
    if len(testing_df) == 0:
        raise ValueError("The DataFrame is empty")
    if not decision_tree:
        raise ValueError("The decision tree is empty")
    correct_count, incorrect_count, unknown_count = 0, 0, 0
    # Classify each row of the DataFrame
    for index, row in testing_df.iterrows():
        classification = classify_row(row, decision_tree)
        if classification is None:
            unknown_count += 1
        elif classification is True:
            correct_count += 1
        else:
            incorrect_count += 1
    ratio = 100/(incorrect_count + correct_count + unknown_count)
    accuracy = correct_count*ratio

    print("TESTING STARTED============")
    print("Number of testing examples", len(testing_df))
    print("Number of testing examples classified = ", correct_count+incorrect_count)
    print("Number of testing examples not classified = ", unknown_count)
    print("Number of correct classification = ", correct_count)
    print("Number of incorrect classification = ", correct_count)
    print("The accuracy is : ", accuracy, "%")
    print("============TESTING ENDED")


training_data = pd.read_csv("C:/Users/fxkik/OneDrive/Documents/CS stuffs/Bachelor stuffs/CS 460 ML/Assignments/Assignment2/data/census_training.csv")
testing_data = pd.read_csv("C:/Users/fxkik/OneDrive/Documents/CS stuffs/Bachelor stuffs/CS 460 ML/Assignments/Assignment2/data/census_training_test.csv")

# get model
decision_tree_model = ID3.build_tree(training_data)
# print accurracy
calculate_accuracy(testing_data, decision_tree_model)
