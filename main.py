from recommendation_system import RecommendationSystem
from utils.load_data import load_data


def main():
    data_path = './data/ratings.csv'
    item_based_matrix_path = './matrix/item_based_matrix.csv'
    user_based_matrix_path = './matrix/user_based_matrix.csv'
    user_label = 'userId'
    item_label = 'movieId'
    score_label = 'rating'
    test_split = 0.15
    min_user_count = 100
    min_item_count = 100
    train_data, test_data = load_data(data_path, user_label, item_label, score_label, test_split,
                                      min_user_count=min_user_count,
                                      min_item_count=min_item_count)
    item_based_model = RecommendationSystem('item_based')
    user_based_model = RecommendationSystem('user_based')

    item_based_model.fit(train_data, user_label, item_label, score_label)
    print('Item_based recommendation system test mean absolute error: {}'.format(item_based_model.test(test_data)))
    item_based_model.save_matrix(item_based_matrix_path)

    user_based_model.fit(train_data, user_label, item_label, score_label)
    print('User_based recommendation system test mean absolute error: {}'.format(user_based_model.test(test_data)))
    user_based_model.save_matrix(user_based_matrix_path)


if __name__ == '__main__':
    main()
