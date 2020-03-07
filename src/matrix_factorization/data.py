import numpy as np
import sklearn.model_selection


class TemporalData:
    def __init__(self, data, ground_truth=None):
        self.X = np.array(data)
        self.N = self.X.shape[0]
        self.T = self.X.shape[1]

        if not(ground_truth is None):
            self.ground_truth = ground_truth


class TemporalDataPrediction(TemporalData):
    def __init__(self, data, ground_truth=None, prediction_rule='last_observed', prediction_window=4):
        super(TemporalDataPrediction, self).__init__(data, ground_truth)

        self.prediction_rule = prediction_rule
        self.prediction_window = 4

        self.valid_rows = np.ones(self.X.shape[0], dtype=np.bool)

        # Find time of last observed entry for all rows
        if self.prediction_rule == 'last_observed':
            time_of_prediction = self.X.shape[1] - np.argmax(
                self.X[:, ::-1] != 0, axis=1) - 1

        # Find time as nearest (in abs. value) nonzero intro to random integer
        elif self.prediction_rule == 'random':
            time_of_prediction = np.array([np.argmin(np.abs(np.random.randint(0, self.X.shape[1]) - np.argwhere(x != 0))) for x in self.X], dtype=np.int)

        # Find rows that permit the specified prediction window
        rows_satisfy_prediction_window = (time_of_prediction > prediction_window)
        self.valid_rows[~rows_satisfy_prediction_window] = False

        # Find rows that have at least one observation before start of prediction window
        rows_satisfy_one_observation = np.argmax(self.X != 0, axis=1) < time_of_prediction - prediction_window
        self.valid_rows[~rows_satisfy_one_observation] = False

        # Remove all rows that don't satisfy the specified criteria
        self.X = self.X[self.valid_rows]
        self.time_of_prediction = time_of_prediction[self.valid_rows]

        # Copy values to be predicted
        self.y = np.copy(self.X[range(self.X.shape[0]), self.time_of_prediction])

        # Overwrite value to be predicted and future values in the regressor dataset
        for i_row in range(self.X.shape[0]):
            self.X[i_row, self.time_of_prediction[i_row]:] = 0


class TemporalDataKFold(TemporalData):
    def __init__(self, data, ground_truth=None, prediction_rule='last_observed', prediction_window=4, n_splits=5):

        if not(ground_truth is None):
            self.ground_truth = ground_truth

        self.prediction_rule = prediction_rule
        self.prediction_window = prediction_window
        self.n_splits = n_splits

        self.__pred_obj = TemporalDataPrediction(data=data, prediction_rule=self.prediction_rule, prediction_window=self.prediction_window)
        self.__train_obj = TemporalData(data=data[self.__pred_obj.valid_rows])

        kf = sklearn.model_selection.KFold(n_splits, shuffle=False)
        self.__idc_per_fold = [idc_fold for idc_fold in kf.split(self.__train_obj.X)]
        
        #Instantiate with 1st fold
        self.__i_fold = 0
        self.__idc_train, self.__idc_pred = self.__idc_per_fold[self.i_fold]
        
    @property
    def X_train(self):
        return self.__train_obj.X[self.__idc_train]

    @property
    def X_pred_regressor(self):
        return self.__pred_obj.X[self.__idc_pred]

    @property
    def y_pred(self):
        return self.__pred_obj.y[self.__idc_pred]

    @property
    def time_of_prediction(self):
        return self.__pred_obj.time_of_prediction[self.__idc_pred]

    @property
    def ground_truth_train(self):
        if (self.ground_truth is None):
            return None 
        return self.ground_truth[self.__idc_train]

    @property
    def ground_truth_pred(self):
        if (self.ground_truth is None):
            return None 
        return self.ground_truth[self.__idc_pred]

    @property
    def i_fold(self):
        return self.__i_fold

    @i_fold.setter
    def i_fold(self, i_fold):
        assert(int(i_fold) < self.n_splits)
        self.__i_fold = int(i_fold)
        self.__idc_train, self.__idc_pred = self.__idc_per_fold[self.__i_fold]