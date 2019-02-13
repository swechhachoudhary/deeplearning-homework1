import numpy as np
import pickle
from relu import Relu
from scipy.misc import logsumexp
from linearLayer import Linear
import matplotlib.pyplot as plt


class NN(object):

    def __init__(self, hidden_dims=(512, 1024), n_hidden=2, para_init="Zero", mode='train', model_path=None):
        self.hidden_dims = hidden_dims
        self.n_hidden = n_hidden
        self.mode = mode
        self.model_path = model_path
        self.input_size = 784
        self.n_class = 10

        self.linear_layers = []

        self.activations = []
        self.relu = Relu()
        self.linear_layers.append(
            Linear(self.input_size, self.hidden_dims[0], parameter_init=para_init))
        for i in range(n_hidden - 1):
            self.linear_layers.append(
                Linear(self.hidden_dims[i], self.hidden_dims[i + 1], parameter_init=para_init))

        self.linear_layers.append(
            Linear(self.hidden_dims[-1], self.n_class, parameter_init=para_init))

    def forward(self, input):
        for i in range(self.n_hidden):
            hidden_output = self.linear_layers[i].forward(input)
            activation = self.relu.relu(hidden_output)
            self.activations.append(activation)
            input = activation
        output = self.linear_layers[-1].forward(input)
        # output = self.softmax(output)
        return output

    def loss(self, true_labels, prediction):
        n_total = len(true_labels)
        # sum of loss over all data samples
        loss = np.sum(-prediction[np.arange(n_total),
                                  np.int_(true_labels)] + logsumexp(prediction, axis=1))

        return loss

    def softmax(self, input, dim=1):
        max_vec = input.max(axis=1, keepdims=True)
        exp_input = np.exp(input - max_vec)
        sum_exp = np.sum(exp_input, axis=1, keepdims=True)
        inp_softmax = exp_input / sum_exp
        return inp_softmax

    def backward(self, input, true_labels, prediction):
        L = self.n_hidden
        n_total = len(true_labels)
        onehot = np.zeros((n_total, 10))
        onehot[np.arange(n_total), true_labels] = 1
        grad_y = []
        grad_a = []
        grad_b = []
        grad_w = []

        # softmax of prediction
        prediction = self.softmax(prediction)

        # gradient of loss with respect to y3(output)

        grad_yL = -onehot + prediction
        grad_y.append(grad_yL)

        for l in range(L, -1, -1):  # 2,1,0 for L = 2

            if l < L:
                grad_al = np.matmul(
                    grad_y[-1], self.linear_layers[l + 1].weights)  # N * h2
                grad_a.append(grad_al)

                grad_yl = grad_al * \
                    self.relu.derivative(self.activations[l])
                grad_y.append(grad_yl)

            grad_bl = grad_y[-1]
            grad_b.append(grad_bl)

            if l == 0:
                grad_wl = np.matmul(grad_y[-1].T, input)
            else:
                grad_wl = np.matmul(
                    grad_y[-1].T, self.activations[l - 1])

            grad_w.append(grad_wl)

        for i in range(L + 1):
            # grad_y[i] = np.sum(grad_y[i], axis=0)
            # grad_a[i] = np.sum(grad_a[i], axis=0)
            grad_b[i] = np.sum(grad_b[i], axis=0, keepdims=True).T

        return grad_w, grad_b

    def update(self, grad_w, grad_b, learning_rate):
        for i in range(self.n_hidden + 1):
            self.linear_layers[-(i + 1)].weights -= learning_rate * grad_w[i]
            self.linear_layers[-(i + 1)].bias -= learning_rate * grad_b[i]

    def accuracy(self, true_label, prediction):
        # softmax of prediction
        prediction = self.softmax(prediction)
        # get class label from softmax of prediction
        predicted_label = np.argmax(prediction, axis=1)
        accuracy = np.sum(true_label == predicted_label) / len(true_label)
        return accuracy

    def train(self, train_data, train_labels, valid_data, valid_labels, batch_size, learning_rate, n_epoch):
        n_total = len(train_labels)
        train_losses = []
        valid_losses = []
        train_accuracy = []
        valid_accuracy = []
        best_valid_acc = 0.0
        best_model_epoch = 0

        # patience p
        p = 7
        k = 0
        for epoch in range(n_epoch):
            start = 0
            end = batch_size
            epoch_loss = 0.0
            train_acc = 0.0
            while end <= n_total:
                # mini-batch
                data = train_data[start: end]
                label = train_labels[start: end]
                # forward pass
                prediction = self.forward(data)

                # compute loss
                batch_loss = self.loss(label, prediction)

                epoch_loss += batch_loss

                # backward pass
                grad_w, grad_b = self.backward(data, label, prediction)

                # parameter update
                self.update(grad_w, grad_b, learning_rate)

                batch_acc = self.accuracy(label, prediction)
                train_acc += batch_acc * batch_size

                start = end
                end = end + batch_size

            avg_epoch_loss = epoch_loss / n_total
            train_losses.append(avg_epoch_loss)

            train_acc = train_acc / n_total
            train_accuracy.append(train_acc)
            print(f'Epoch {epoch} train loss : {avg_epoch_loss}')
            print(f'Epoch {epoch} train accuracy : {train_acc}')

            valid_loss, valid_acc = self.validate(
                valid_data, valid_labels, batch_size)

            print(f'Epoch {epoch} validation loss : {valid_loss}')
            print(f'Epoch {epoch} validation accuracy : {valid_acc}')
            valid_losses.append(valid_loss)
            valid_accuracy.append(valid_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_model_epoch = epoch
                print(f'Best validation performance at epoch {epoch} with validation accuracy : {valid_acc}')
                # save the best model
                with open('model.pickle', 'wb') as f:
                    pickle.dump(self.linear_layers, f)
                k = 0
            # elif k == p:
            #     break
            # else:
            #     k += 1
            if not (epoch + 1) % 25:
                learning_rate /= 2
                print(f'At Epoch {epoch}  learning rate reduced to : {learning_rate}')
        plt.plot(train_losses, label="Train loss")
        plt.plot(valid_losses, label="valid loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("learning_curves.png")
        return

    def validate(self, valid_data, valid_labels, batch_size):
        n_valid = len(valid_labels)
        start = 0
        end = batch_size
        valid_loss = 0.0
        valid_acc = 0.0
        while end <= n_valid:
            # mini-batch
            data = valid_data[start: end]
            label = valid_labels[start: end]
            # forward pass
            prediction = self.forward(data)

            # compute loss
            batch_loss = self.loss(label, prediction)

            valid_loss += batch_loss

            batch_acc = self.accuracy(label, prediction)
            valid_acc += batch_acc * batch_size

            start = end
            end = end + batch_size

        avg_valid_loss = valid_loss / n_valid
        avg_valid_acc = valid_acc / n_valid

        return avg_valid_loss, avg_valid_acc
