import numpy as np
import sklearn.model_selection


class TemporalData:
    def __init__(self, data):
        self.X = np.array(data)
        self.N = self.X.shape[0]
        self.T = self.X.shape[1]


class TemporalDataPrediction(TemporalData):
    def __init__(self, data, time_of_prediction):
        super(TemporalDataPrediction, self).__init__(data)
        self.time_of_prediction = time_of_prediction

        if 'time_of_prediction' == 'last_observed':
            time_of_last_observation_X = self.X.shape[1] - np.argmax(
                self.X[:, ::-1] != 0, axis=1) - 1
            self.y = np.copy(self.X[range(self.X.shape[0]), time_of_last_observation_X]
                             )
            self.X[range(
                self.X.shape[0]), time_of_last_observation_X] = 0


class TemporalDataKFold(TemporalData):
    def __init__(self, data, time_of_prediction, n_splits=5):
        super(TemporalDataKFold, self).__init__(data)
        self.time_of_prediction = time_of_prediction

        kf = sklearn.model_selection.KFold(n_splits, shuffle=False)
        self.fold_indices = [idc for idc in kf.split(self.X)]

    def get_fold(self, k):
        train_indices, pred_indices = self.fold_indices[k]

        X_train = self.X[train_indices]
        X_pred = self.X[pred_indices]

        if self.time_of_prediction == 'last_observed':
            time_of_last_observation_X_pred = X_pred.shape[1] - np.argmax(
                X_pred[:, ::-1] != 0, axis=1) - 1
            y_pred = X_pred[range(X_pred.shape[0]),
                            time_of_last_observation_X_pred]

            X_pred_regressor = np.copy(X_pred)
            X_pred_regressor[range(X_pred.shape[0]),
                             time_of_last_observation_X_pred] = 0

            return X_train, X_pred_regressor, y_pred, time_of_last_observation_X_pred
