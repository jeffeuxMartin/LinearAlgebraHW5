from typing import Tuple

import matplotlib.pyplot as plt

# functions and classes from `lg.py`
from lg import read_train_csv, read_test_csv
from lg import LinearRegression, MSE


def run_train_test(
    N: int,
    train_filename: str,
    test_filename: str
  ) -> Tuple[float, float]:
    """
    Running the training and testing process.
    Returning the two losses.
    """

    train_X, train_Y = read_train_csv(
        train_filename, N)
    model = LinearRegression()
    model.train(train_X, train_Y)
    pred_train_Y = model.predict(train_X)
    train_loss = MSE(pred_train_Y, train_Y)

    test_X, test_Y = read_test_csv(
        test_filename, N, "val" in test_filename)
    pred_test_Y = model.predict(test_X)
    test_loss = MSE(pred_test_Y, test_Y)
    print(train_loss, test_loss)

    return train_loss, test_loss


def plot_N_loss(
    train_filename:   str = "data/train.csv",
    test_filename:    str = "data/validation.csv",
    savefig_filename: str = "result.png",
  ):
    """
    Plot the loss after running `run_train_test`.
    """

    all_train_losses = []
    all_test_losses = []
    for N in range(1, 31):
        train_loss, test_loss = (
            run_train_test(
                N, train_filename, test_filename))
        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)

    N = list(range(1, 31))
    plt.plot(N, all_train_losses, "b",
             label="train")
    plt.plot(N, all_test_losses,  "r",
             label="validation")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("MSE loss")
    plt.savefig(savefig_filename)


if __name__ == "__main__" :
    plot_N_loss()

    # TODO: Please change this to the `N` that you
    #       would like to use.
    N = 0
    assert N != 0, "`N` should be changed!"
    run_train_test(N,
                   train_filename="data/train.csv",
                   test_filename="data/test.csv")
