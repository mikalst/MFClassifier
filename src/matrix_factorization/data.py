import numpy as np
import sklearn.model_selection


class TemporalData:
    def __init__(self, data):
        self.X = np.array(data)
        self.N = self.X.shape[0]
        self.T = self.X.shape[1]


class TemporalDataPrediction(TemporalData):
    def __init__(self, data, prediction_rule='last_observed', prediction_window=4):
        super(TemporalDataPrediction, self).__init__(data)

        self.prediction_rule = prediction_rule

        valid_rows = np.ones(self.X.shape[0], dtype=np.bool)

        # Find time of last observed entry for all rows
        if self.prediction_rule == 'last_observed':
            time_of_prediction = self.X.shape[1] - np.argmax(
                self.X[:, ::-1] != 0, axis=1) - 1

        # Find time as nearest (in abs. value) nonzero intro to random integer
        elif self.prediction_rule == 'random':
            time_of_prediction = np.array([np.argmin(np.abs(np.random.randint(0, self.X.shape[1]) - np.argwhere(x != 0))) for x in self.X], dtype=np.int)

        # Find rows that permit the specified prediction window
        rows_satisfy_prediction_window = (time_of_prediction > prediction_window)
        valid_rows[~rows_satisfy_prediction_window] = False

        # Find rows that have at least one observation before start of prediction window
        rows_satisfy_one_observation = np.argmax(self.X != 0, axis=1) < time_of_prediction - prediction_window
        valid_rows[~rows_satisfy_one_observation] = False

        # Remove all rows that don't satisfy the specified criteria
        self.X = self.X[valid_rows]
        self.time_of_prediction = time_of_prediction[valid_rows]

        # Copy values to be predicted
        self.y = np.copy(self.X[range(self.X.shape[0]), self.time_of_prediction])

        # Overwrite value to be predicted and future values in the regressor dataset
        for i_row in range(self.X.shape[0]):
            self.X[i_row, self.time_of_prediction[i_row]:] = 0


class TemporalDataKFold(TemporalData):
    def __init__(self, data, prediction_rule, prediction_window=4, n_splits=5):
        super(TemporalDataKFold, self).__init__(data)
        
        self.prediction_rule = prediction_rule
        self.prediction_window = prediction_window

        self.pred_obj = TemporalDataPrediction(data=self.X, prediction_rule=self.prediction_rule, prediction_window=self.prediction_window)

        kf = sklearn.model_selection.KFold(n_splits, shuffle=False)
        self.fold_indices = [idc for idc in kf.split(self.X)]

    def get_fold_idc(self, k):
        return self.fold_indices[k]

    def get_fold(self, k):
        train_indices, pred_indices = self.fold_indices[k]

        return self.X[train_indices], self.pred_obj.X[pred_indices], self.pred_obj.y[pred_indices], self.pred_obj.time_of_prediction[pred_indices]