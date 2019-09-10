import pandas as pd


def split_train_test_data(data, test_split):
    # randomize data
    data = data.sample(frac=1.0)
    index = int(data.index.size * (1 - test_split))
    train_data = data[:index]
    test_data = data[index:]
    return train_data, test_data


def load_data(file_path, user_label, item_label, score_label, test_split=0.15, min_item_count=0, min_user_count=0):
    data = pd.read_csv(file_path)
    data = data[[user_label, item_label, score_label]]
    data.dropna(inplace=True)

    # drop rating data that having not enough count
    counts1 = data[user_label].value_counts()
    data = data[data[user_label].isin(counts1[counts1 >= min_user_count].index)]
    counts2 = data[item_label].value_counts()
    data = data[data[item_label].isin(counts2[counts2 >= min_item_count].index)]

    # normalize score values to range [0, 1]
    scores = data[score_label].values
    data[score_label] = (scores - scores.min()) / (scores.max() - scores.min())
    train_data, test_data = split_train_test_data(data, test_split)
    return train_data, test_data
