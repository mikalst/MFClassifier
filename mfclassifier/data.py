import numpy as np
import sklearn.model_selection


class _TemporalDataset:
    def __init__(self, data):
        self.X = np.array(data)


class TemporalDatasetTrain(_TemporalDataset):
    def __init__(self, data, ground_truth=None):
        super(TemporalDatasetTrain, self).__init__(data)
        self.ground_truth = np.array(ground_truth)

    @property
    def X_train(self):
        return self.X

    @property
    def ground_truth_train(self):
        return self.ground_truth


class TemporalDatasetPredict(_TemporalDataset):
    def __init__(self, data, ground_truth=None, prediction_rule='last_observed', prediction_window=4, random_state=42):
        super(TemporalDatasetPredict, self).__init__(data)

        self.prediction_rule = prediction_rule
        self.prediction_window = 4

        # Find time of last observed entry for all rows
        if self.prediction_rule == 'last_observed':
            time_of_prediction = self.X.shape[1] - np.argmax(
                self.X[:, ::-1] != 0, axis=1) - 1

        # Find time as nearest (in abs. value) nonzero intro to random integer
        elif self.prediction_rule == 'random':
            np.random.seed(random_state)
            time_of_prediction = np.array([np.argmin(np.abs(np.random.randint(
                0, self.X.shape[1]) - np.arange(0, self.X.shape[1])*(x != 0))) for x in self.X], dtype=np.int)

        # Copy values to be predicted
        y_true = np.copy(self.X[range(self.X.shape[0]), time_of_prediction])

        # Remove observations in or after prediction window
        for i_row in range(self.X.shape[0]):
            self.X[i_row, max(0, time_of_prediction[i_row] -
                              prediction_window):] = 0

        # Find rows that still contain observations
        self.valid_rows = np.sum(self.X, axis=1) > 0

        # Remove all rows that don't satisfy the specified criteria
        self.y = y_true[self.valid_rows]
        self.X = self.X[self.valid_rows]
        self.time_of_prediction = time_of_prediction[self.valid_rows]

        self.ground_truth = ground_truth
        if not(self.ground_truth is None):
            self.ground_truth = ground_truth[self.valid_rows]

    @property
    def X_pred_regressor(self):
        return self.X

    @property
    def y_true(self):
        return self.y

    @property
    def ground_truth_pred(self):
        return self.ground_truth


class TemporalDatasetKFold(_TemporalDataset):
    def __init__(self, data, ground_truth=None, prediction_rule='last_observed', prediction_window=4, n_splits=5, random_state=42):
        self.prediction_rule = prediction_rule
        self.prediction_window = prediction_window
        self.n_splits = n_splits

        self.__pred_obj = TemporalDatasetPredict(
            data=data, ground_truth=ground_truth, prediction_rule=self.prediction_rule, prediction_window=self.prediction_window, random_state=random_state)
        if ground_truth is None:
            self.__train_obj = TemporalDatasetTrain(
                data=data[self.__pred_obj.valid_rows])
        else:
            self.__train_obj = TemporalDatasetTrain(
                data=data[self.__pred_obj.valid_rows], ground_truth=ground_truth[self.__pred_obj.valid_rows])

        kf = sklearn.model_selection.KFold(n_splits, shuffle=False)
        self.__idc_per_fold = [
            idc_fold for idc_fold in kf.split(self.__train_obj.X)]

        # Instantiate with 1st fold
        self.__i_fold = 0
        self.__idc_train, self.__idc_pred = self.__idc_per_fold[self.i_fold]

    @property
    def X_train(self):
        return self.__train_obj.X[self.__idc_train]

    @property
    def X_pred_regressor(self):
        return self.__pred_obj.X[self.__idc_pred]

    @property
    def y_true(self):
        return self.__pred_obj.y[self.__idc_pred]

    @property
    def time_of_prediction(self):
        return self.__pred_obj.time_of_prediction[self.__idc_pred]

    @property
    def ground_truth_train(self):
        if (self.__train_obj.ground_truth is None):
            return None
        return self.__train_obj.ground_truth[self.__idc_train]

    @property
    def ground_truth_pred(self):
        if (self.__pred_obj.ground_truth is None):
            return None
        return self.__pred_obj.ground_truth[self.__idc_pred]

    @property
    def i_fold(self):
        return self.__i_fold

    @i_fold.setter
    def i_fold(self, i_fold):
        assert(int(i_fold) < self.n_splits)
        self.__i_fold = int(i_fold)
        self.__idc_train, self.__idc_pred = self.__idc_per_fold[self.__i_fold]
