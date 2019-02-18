import _pickle
import gzip
import numpy as np
import argparse
import logging
import pickle
import matplotlib.pyplot as plt
from neuralNetwork import NN

np.random.seed(111)


def arg_parser():
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_hidden", type=int, default=2)
    argparser.add_argument("--n_epoch", type=int, default=70)
    argparser.add_argument("--batch_size", type=int, default=50)
    argparser.add_argument("--learning_rate", type=float, default=0.001)
    argparser.add_argument("--para_init", type=str, default="Glorot")
    args = argparser.parse_args()

    return args


def data_loader():
    # Load the dataset
    f = gzip.open("data/mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = _pickle.load(f, encoding="latin1")
    f.close()

    # 50,000 images
    train_data = train_set[0]
    train_label = train_set[1]

    # 10,000 images
    valid_data = valid_set[0]
    valid_label = valid_set[1]

    # 10,000 images
    test_data = test_set[0]
    test_label = test_set[1]

    return train_data, train_label, valid_data, valid_label, test_data, test_label


def using_different_initialization(
    train_data, train_label, batch_size, learning_rate, n_epoch=10
):
    n_total = len(train_label)

    logging.basicConfig(
        filename="using_different_initialization.log", level=logging.INFO
    )

    para_inits = ["Zero", "Normal", "Glorot"]

    for para_init in para_inits:

        logging.info(
            f"-------------train loss for {para_init} initialization of weights------------"
        )

        model = NN(hidden_dims=(512, 1024), n_hidden=2, para_init=para_init)
        train_losses = []
        for epoch in range(n_epoch):
            start = 0
            end = batch_size
            epoch_loss = 0.0
            while end <= n_total:
                # mini-batch
                data = train_data[start:end]
                label = train_label[start:end]
                # forward pass
                prediction = model.forward(data)

                # compute loss
                batch_loss = model.loss(label, prediction)

                epoch_loss += batch_loss

                # backward pass
                grad_w, grad_b = model.backward(data, label, prediction)

                # parameter update
                model.update(grad_w, grad_b, learning_rate)

                start = end
                end = end + batch_size

            avg_epoch_loss = epoch_loss / n_total
            train_losses.append(avg_epoch_loss)

            logging.info(
                f"Epoch {epoch} train loss for {para_init} initialization of weights: {avg_epoch_loss}"
            )
        plt.plot(train_losses, label=para_init)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("que1_results/train_loss_curves.png")
    plt.close()

# Plot the maximum difference between the true gradient and the finite
# difference gradient as a function of N where epsilon=1/N


def plot_max_diff_between_gradients(
        train_data, train_label, model):

    logging.basicConfig(
        filename="finite_difference_gradient.log", level=logging.INFO
    )

    sample_index = np.random.randint(0, high=len(train_label))
    data = train_data[sample_index: sample_index + 1]
    label = train_label[sample_index: sample_index + 1]
    logging.info(
        f"------Sample data index for which gradient is computed : {sample_index}------"
    )
    # forward pass
    prediction = model.forward(data)

    # compute loss
    loss = model.loss(label, prediction)

    # backward pass for true gradient
    grad_w, grad_b = model.backward(data, label, prediction)

    p = np.minimum(10, 512 * 784)

    N = np.array([50, 500, 1000, 10000, 100000])

    epsilons = 1.0 / N

    max_difference = []

    for epsilon in epsilons:
        logging.info(
            f"------Finite difference gradient for first {p} parameters for epsilon {epsilon}------"
        )
        finite_grad = []

        for i in range(p):
            model.linear_layers[0].weights[511, 546 + i] += epsilon

            # forward pass
            prediction = model.forward(data)

            # compute loss
            loss1 = model.loss(label, prediction)

            model.linear_layers[0].weights[511, 546 + i] -= 2 * epsilon

            # forward pass
            prediction = model.forward(data)

            # compute loss
            loss2 = model.loss(label, prediction)

            gradient = (loss1 - loss2) / (2 * epsilon)
            finite_grad.append(gradient)

            # reset parameter to previous values
            model.linear_layers[0].weights[511, 546 + i] += epsilon

        true_grad = grad_w[-1][511, 546: 546 + p]
        logging.info(f"True gradient : {true_grad}")
        logging.info(f"Finite difference gradient : {finite_grad}")

        max_diff = np.max(np.abs(finite_grad - true_grad))
        max_difference.append(max_diff)

    logging.info(f"Maximum gradient difference: {max_difference}")
    plt.semilogx(N, max_difference)
    plt.xlabel("N")
    plt.ylabel("max difference")
    plt.savefig("max_difference_curve.png")
    plt.close()


def test(test_data, test_labels, batch_size, model):
    n_test = len(test_labels)
    start = 0
    end = batch_size
    test_loss = 0.0
    test_acc = 0.0
    while end <= n_test:
        # mini-batch
        data = test_data[start:end]
        label = test_labels[start:end]
        # forward pass
        prediction = model.forward(data)

        # compute loss
        batch_loss = model.loss(label, prediction)

        test_loss += batch_loss

        batch_acc = model.accuracy(label, prediction)
        test_acc += batch_acc * batch_size

        start = end
        end = end + batch_size

    avg_test_loss = test_loss / n_test
    avg_test_acc = test_acc / n_test

    print(f"Test loss : {avg_test_loss}")
    print(f"Test accuracy : {avg_test_acc}")


if __name__ == "__main__":
    # Load the dataset
    train_data, train_label, valid_data, valid_label, test_data, test_label = data_loader()

    args = arg_parser()

    # part 1 of question 1
    using_different_initialization(
        train_data, train_label, args.batch_size, args.learning_rate,
        n_epoch=10)

    model = NN(
        hidden_dims=(512, 1024), n_hidden=args.n_hidden, para_init=args.para_init
    )

    model.train(
        train_data,
        train_label,
        valid_data,
        valid_label,
        args.batch_size,
        args.learning_rate,
        args.n_epoch,
    )

    # # load best model
    with open("model.pickle", "rb") as f:
        best_parameters = pickle.load(f)

    best_model = NN(
        hidden_dims=(512, 1024), n_hidden=args.n_hidden, para_init=args.para_init
    )

    best_model.linear_layers = best_parameters

    test(test_data, test_label, args.batch_size, best_model)

    plot_max_diff_between_gradients(train_data, train_label, best_model)
