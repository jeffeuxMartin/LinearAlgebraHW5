from typing import Tuple

import numpy as np

attrs = ["AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2",
         "NOx", "O3", "PM10",
         "PM2.5",  # <-- The target feature
         "RAINFALL", "RH", "SO2", "THC", "WD_HR",
         "WIND_DIREC", "WIND_SPEED", "WS_HR"]
DAYS = np.array([31, 28, 31, 30, 31, 30,
                 31, 31, 30, 31, 30, 31])

def get_N_hours_feat(
    month_data: np.ndarray, N: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    """ Get features of N hours. """

    # month_data.shape = (num_of_date, 18, 24)

    data = (month_data.transpose((0, 2, 1))
                      .reshape(-1, 18))
    label = (month_data.transpose((1, 0, 2))
                       .reshape(18, -1))[9]
    total_hours = len(label)

    feats = np.array([])
    for i in range(total_hours - N):
        # Adding `w0`. To discuss without `w0`,
        #     please comment the next command!
        #     (split into 2 lines by default)-----|
        cur_feat = np.append(                # <--|
            data[i : i + N].flatten(), [1])  # <--|
        feats = np.concatenate(
            [feats, cur_feat], axis=0)

    label = label[N:]

    # To discuss without `w0`,
    #     please change `N * 18 + 1` to `N * 18`!
    feats = feats.reshape(-1, N * 18 + 1)

    return feats, label


def read_train_csv(
    fileName: str, N: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    A utility function for reading the training data
    in CSV format.
    """

    data = np.genfromtxt(fileName,
                         delimiter=',',
                         skip_header=1
                        )[:, 3:].astype(float)
    # 12 months, 20 days per month,
    #     18 features per day, 24 hours per day
    data = data.reshape(12, -1, 18, 24)
    train_X, train_Y = get_N_hours_feat(data[0], N)

    for i in range(1, 12):
        X, Y = get_N_hours_feat(data[i], N)
        train_X = np.concatenate((train_X, X), axis=0)
        train_Y = np.concatenate((train_Y, Y), axis=0)

    return train_X, train_Y


def read_test_csv(
    fileName: str, N: int, isval: bool
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    A utility function for reading the testing data
    in CSV format. `isval` is for validation.
    """

    if not isval:
        test_days = DAYS - 22
        cumul_days = [sum(test_days[:i])
                          for i in range(1, 12 + 1)]
    else:
        # test_days = 5
        cumul_days = [2 * i for i in range(1, 12 + 1)]
    data = (np.genfromtxt(fileName,
                          delimiter=',',
                          skip_header=1
                         )[:, 3:].astype(float)
                                 .reshape(-1, 18, 24))

    test_X, test_Y = get_N_hours_feat(
        data[:cumul_days[0]], N)

    for i in range(1, 12):
        X, Y = get_N_hours_feat(
            data[cumul_days[i - 1] : cumul_days[i]],
            N)
        test_X = np.concatenate((test_X, X), axis=0)
        test_Y = np.concatenate((test_Y, Y), axis=0)

    return test_X, test_Y


class LinearRegression(object):
    """
    A class wrapper for linear regression.

    Attributes
    ----------
    W : np.ndarray
        The weight vector for linear regression.

    Methods
    -------
    train(X, Y):
        Training the regressor by X and Y.
    predict(X):
        Predict from X and the W vector.
    """

    def __init__(self): pass

    def train(self, X: np.ndarray, Y: np.ndarray):
        """ Input X and Y to train the W vector. """
        # TODO: the shape of W should be
        #       number of features
        W = np.array()
        self.W = W

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Predict by the given X and trained W. """
        # TODO
        pred_X = np.array()
        return pred_X


def MSE(
  pred_Y: np.ndarray, real_Y: np.ndarray
) -> float:
    """ Return the MSE by predicted and real data. """
    # TODO: mean square error
    error: float = 0.
    return error
