import numpy as np
import ot
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute optimal transport distance

def OT(x, y):
    assert x.shape[0] == y.shape[0], "The number of samples in x and y must be the same"
    assert x.shape[1] == y.shape[1], "The dimension of x and y must be the same"

    n = x.shape[0]
    a, b = torch.ones(n).to(device) / n, torch.ones(n).to(device) / n

    C = ot.dist(y, x)
    scaling = C.max()
    C /= scaling
    C = torch.tensor(C, dtype = torch.float32).to(device)

    G0 = ot.emd(a, b, C, numItermax = 10_000_000)

    return scaling * (G0 * C).sum(), G0

# Display transport plan

def arrow(X, Y):
    plt.arrow(X[0], X[1], Y[0] - X[0], Y[1] - X[1], color = [0.4, 0.4, 0.9], alpha = 0.1)

plotp = lambda x,col, alpha=1.0: plt.scatter(x[0,:], x[1,:], s=200, edgecolors="k", c=col, linewidths=2, alpha=alpha)

def display_transport_plan(X, Y, X_Y, Y_X, step, POT_distance, approx_distance):
    x_min, x_max, y_min, y_max = X[0, :].min() - 2, X[0, :].max() + 2, X[1, :].min() - 2, X[1, :].max() + 2

    width = 12
    height = width * (y_max - y_min) / (x_max - x_min)
    height = 6

    plt.figure(figsize = (width, height))
    plt.suptitle(f"Step: {step} | POT distance: {POT_distance:.3f} | ICNN distance: {approx_distance:.3f}")

    plt.subplot(1, 2, 1)
    plt.scatter(X[0, :], X[1, :], s = 50, edgecolors = "k", linewidths = 2)
    plt.scatter(Y[0, :], Y[1, :], s = 50, edgecolors = "k", linewidths = 2)
    plt.scatter(Y_X[0, :], Y_X[1, :], s = 50, edgecolors = "k", linewidths = 2, alpha = 0.2)
    for i in range(Y.shape[1]):
        arrow(Y[:, i], Y_X[:, i])
    plt.legend(["X", "Y", r"$Y \to X$"])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    plt.scatter(X[0, :], X[1, :], s = 50, edgecolors = "k", linewidths = 2)
    plt.scatter(Y[0, :], Y[1, :], s = 50, edgecolors = "k", linewidths = 2)
    plt.scatter(X_Y[0, :], X_Y[1, :], s = 50, edgecolors = "k", linewidths = 2, alpha = 0.2)
    for i in range(X.shape[1]):
        arrow(X[:, i], X_Y[:, i])
    plt.legend([r"$\mu$", r"$\nu$", r"$\mu \to \nu$"])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis("equal")

    plt.savefig(f"output/transport_plan_{step}.png")
    plt.close()