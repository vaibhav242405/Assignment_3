import torch
from torch import nn
import math


def create_linear_regression_model(input_size: int, output_size: int):
    """
    Create a linear regression model with the given input and output sizes.
    Hint: use nn.Linear
    Args:
        input_size: int, in_features (int), size of each input sample
        output_size: int, out_features (int), size of each output sample
    """
    model = nn.Linear(in_features=input_size, out_features=output_size)

    return model


def train_iteration(X: torch.Tensor, y: torch.Tensor, model, loss_fn, optimizer):
    """
    Compute the prediction, Calculate the gradients, and return loss
    Args:
        X: torch.Tensor, input data
        y: torch.Tensor, target data (actual)
        model: 
        loss_fn:
        optimizer:
    Return:
        loss
    """
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation, Clear gradients from previous iteration
    optimizer.zero_grad()
    # Calculate gradients, Gradients now stored in input.grad
    loss.backward()
    # Updates model parameters: It's a crucial function for iteratively adjusting 
    # the model's parameters (weights and biases) during training 
    # to guide it towards better performance.
    optimizer.step()

    return loss


def fit_regression_model(X: torch.Tensor, y: torch.Tensor) -> tuple:
    """
    Train the model for the given number of epochs.
    Hint: use the train_iteration function.
    Hint 2: while woring you can use the print function to print the loss every 1000 epochs.
    Hint 3: you can use the previos_loss variable to stop the training when the loss is not changing much.
    """
    learning_rate = 0.002  # Pick a better learning rate
    num_epochs = 10000  # Pick a better number of epochs
    input_features = X.shape[1]  # extract the number of features from the input `shape` of X
    output_features = y.shape[1]  # extract the number of features from the output `shape` of y
    model = create_linear_regression_model(input_features, output_features)

    # loss_fn = nn.L1Loss() # Use mean squared error loss, like in class
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.09)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    previos_loss, loss = float("inf"), None

    for epoch in range(1, num_epochs):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        curr_loss = float(loss.item())
        # print('Loss %f, Prev Loss %f, Diff %f' % (curr_loss, previos_loss, previos_loss - curr_loss))
        if (not math.isinf(previos_loss)) and (
                abs(previos_loss - curr_loss) < 0.002):  # Change this condition to stop the training when the loss is not changing much.
            break
        previos_loss = curr_loss
        # This is a good place to print the loss every 1000 epochs.
        if (epoch % 1000) == 0:
            print('Epoch %d, Loss %f, Prev Loss %f' % (epoch, curr_loss, previos_loss))

    return model, loss