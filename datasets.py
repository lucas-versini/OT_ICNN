import numpy as np
import torch

def generate_two_gaussians(n_samples = 1000, d = 2):
    """ Generate two gaussians with given parameters """
    mean1 = torch.zeros(d, dtype = torch.float32)
    mean2 = 4 * torch.ones(d, dtype = torch.float32)
    var1 = 1
    var2 = 1

    X = torch.tensor(np.random.multivariate_normal(mean1, var1 * np.eye(d), n_samples), dtype = torch.float32)
    Y = torch.tensor(np.random.multivariate_normal(mean2, var2 * np.eye(d), n_samples), dtype = torch.float32)
    return X, Y

def generate_n_gaussians(n = 2, n_samples = 1000):
    n_per_gaussian = int(n_samples / n)
    X = []
    for i in range(n):
        theta = 2 * np.pi * i / n
        mean = [10 * np.cos(theta), 10 * np.sin(theta)]
        var_ = 1
        X.append(np.random.multivariate_normal(mean, var_ * np.eye(2), n_per_gaussian))
    X = torch.tensor(np.concatenate(X, axis = 0), dtype = torch.float32)

    Y = np.random.multivariate_normal([0, 0], 1 * np.eye(2), size = n * int(n_samples / n))
    Y = torch.tensor(Y, dtype = torch.float32)

    return X, Y

def generate_two_circles(n_samples = 1000, r1 = 1, r2 = 2):
    theta_X = 2 * np.pi * np.random.rand(n_samples)
    r_X = r1 * np.random.rand(n_samples)
    X = np.array([r_X * np.cos(theta_X), r_X * np.sin(theta_X)]).T
    X = X + 0.1 * np.random.randn(n_samples, 2)
    X = torch.tensor(X, dtype = torch.float32)

    theta_Y = 2 * np.pi * np.random.rand(n_samples)
    r_Y = r2 + r1 * np.random.rand(n_samples)
    Y = np.array([r_Y * np.cos(theta_Y), r_Y * np.sin(theta_Y)]).T
    Y = Y + 0.1 * np.random.randn(n_samples, 2)
    Y = torch.tensor(Y, dtype = torch.float32)

    return X, Y

def generate_Voronoi(n_samples):
    min_, max_ = -10, 10

    centers = np.array([[(3 * min_ + max_) / 4, (3 * min_ + max_) / 4],
                        [(3 * min_ + max_) / 4, (min_ + 3 * max_) / 4],
                        [(min_ + 3 * max_) / 4, (3 * min_ + max_) / 4],
                        [(min_ + 3 * max_) / 4, (min_ + 3 * max_) / 4]])
    
    X = (max_ - min_) * np.random.rand(n_samples, 2) + min_
    Y = centers[np.random.randint(0, 4, n_samples)]

    X = torch.tensor(X, dtype = torch.float32)
    Y = torch.tensor(Y, dtype = torch.float32)

    return X, Y

def generate_segments(n_samples):
    min_, max_ = 0, 1

    X = (max_ - min_) * np.random.rand(n_samples, 2) + min_
    Y = (max_ - min_) * np.random.rand(n_samples, 2) + min_
    X[:, 1] = 0.
    Y[:, 0] = 0.

    X = torch.tensor(X, dtype = torch.float32)
    Y = torch.tensor(Y, dtype = torch.float32)

    return X, Y

def generate_dataset(dataset, n = 5000):
    """ Generate a dataset with given parameters """
    if dataset["type"] == "gaussians":
        if dataset["d"] == 2:
            return generate_n_gaussians(n = dataset["n"], n_samples = n)
        else:
            return generate_two_gaussians(n_samples = n, d = dataset["d"])
    elif dataset["type"] == "two_circles":
        return generate_two_circles(n_samples = n)
    elif dataset["type"] == "Voronoi":
        return generate_Voronoi(n)
    elif dataset["type"] == "segments":
        return generate_segments(n)
    else:
        raise ValueError(f"Dataset type {dataset['type']} not recognized")