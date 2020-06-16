import numpy as np
import sklearn.metrics


def confusion_matrix(model, X, t, y_true):
    predicted_z = model.predict(X, t)

    return sklearn.metrics.confusion_matrix(y_true, predicted_z)


def confusion_matrix_bin(model, X, t, y_true):
    predict_bin = model.predict_binary(X, t)
    y_true_bin = model.z_to_binary_mapping(y_true)

    return sklearn.metrics.confusion_matrix(y_true_bin, predict_bin)


def accuracy_score(model, X, t, y_true):
    # Classification full
    predicted_z = model.predict(X, t)

    return np.mean(predicted_z == y_true)


def accuracy_event_score(model, X, t, y_true):
    predicted_e = model.predict_event(X, t)

    return np.mean(predicted_e == list(map(model.z_to_e_mapping, y_true)))


def sensitivity_score(model, X, t, y_true):
    predicted_bin = model.predict_binary(X, t)
    y_true_bin = model.z_to_binary_mapping(y_true)

    truly_positive = y_true_bin == 1

    return np.mean(predicted_bin[truly_positive] == y_true_bin[truly_positive])


def specificity_score(model, X, t, y_true):
    predicted_bin = model.predict_binary(X, t)
    y_true_bin = model.z_to_binary_mapping(y_true)

    truly_negative = y_true_bin == 0

    return np.mean(predicted_bin[truly_negative] == y_true_bin[truly_negative])


def mcc_score(model, X, t, y_true):
    predicted_bin = model.predict_binary(X, t)
    y_true_bin = model.z_to_binary_mapping(y_true)

    tp = np.sum(predicted_bin[y_true_bin == 1] == y_true_bin[y_true_bin == 1])
    fp = np.sum(y_true_bin == 1) - tp

    tn = np.sum(predicted_bin[y_true_bin == 0] == y_true_bin[y_true_bin == 0])
    fn = np.sum(y_true_bin == 0) - tp

    print(tp, fp, tn, fn)

    return (tp*tn - fp*fn)/np.sqrt((tp+fp)*(tp + fn)*(tn + fp)*(tn + fn))
