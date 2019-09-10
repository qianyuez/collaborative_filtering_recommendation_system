import numpy as np
import pandas as pd
from utils.similarity import cosine_similarity


class RecommendationSystem():
    def __init__(self, model_type='item_based'):
        self.model_type = model_type
        self.score_matrix = None

    def fit(self,
            data,
            user_label,
            item_label,
            score_label):
        if self.model_type == 'item_based':
            self.score_matrix = self._build_item_based_matrix(data, user_label, item_label, score_label)
        elif self.model_type == 'user_based':
            self.score_matrix = self._build_user_based_matrix(data, user_label, item_label, score_label)
        else:
            raise ValueError('unknown model type')

    def predict_score(self, user_name, item_name):
        if user_name not in self.score_matrix.index:
            return None
        if item_name not in self.score_matrix.columns:
            return None
        return self.score_matrix.loc[user_name, item_name]

    def save_matrix(self, path):
        self.score_matrix.to_csv(path)

    def load_matrix(self, path):
        self.score_matrix = pd.read_csv(path)

    def test(self, data):
        mae = []
        for _, s in data.iterrows():
            user = s[0]
            item = s[1]
            score = s[2]
            score_ = self.predict_score(user, item)
            if score_ is not None:
                mae.append(np.abs(score - score_))
        return np.mean(mae)

    def _build_item_based_matrix(self, data, user_label, item_label, value_label):
        value_matrix = data.pivot(index=user_label, columns=item_label, values=value_label)
        for item, s in value_matrix.iteritems():
            value_matrix[item].fillna(s.mean(), inplace=True)
        item_similarity_matrix = np.array(value_matrix.corr(method=cosine_similarity))
        item_sp_matrix = np.matmul(np.array(value_matrix), item_similarity_matrix)
        item_sr_matrix = np.sum(item_similarity_matrix, axis=-1).reshape((1, -1))
        item_based_score_matrix = item_sp_matrix / item_sr_matrix
        df = pd.DataFrame(item_based_score_matrix, index=value_matrix.index, columns=value_matrix.columns)
        return df

    def _build_user_based_matrix(self, data, user_label, item_label, value_label):
        value_matrix = data.pivot(index=user_label, columns=item_label, values=value_label)
        for user, s in value_matrix.iterrows():
            value_matrix.loc[user].fillna(s.mean(), inplace=True)
        user_similarity_matrix = np.array(value_matrix.T.corr(method=cosine_similarity))
        user_sp_matrix = np.matmul(user_similarity_matrix, np.array(value_matrix))
        user_sr_matrix = np.sum(user_similarity_matrix, axis=-1).reshape((-1, 1))
        user_based_score_matrix = user_sp_matrix / user_sr_matrix
        df = pd.DataFrame(user_based_score_matrix, index=value_matrix.index, columns=value_matrix.columns)
        return df
