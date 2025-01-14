import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import time

from utils import display_transport_plan, OT
from datasets import generate_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ICNN(nn.Module):
    """ Implementation of an Input Convex Neural Network (ICNN) """
    def __init__(self, d = 2, hidden_size = 64, num_layers = 4, positive = True):
        super(ICNN, self).__init__()
        self.linear_layers = nn.ModuleList()
        self.convex_layers = nn.ModuleList()
        
        self.linear_layers.append(nn.Linear(d, hidden_size, bias = True))
        for _ in range(num_layers - 2):
            self.linear_layers.append(nn.Linear(d, hidden_size, bias = True))
            self.convex_layers.append(nn.Linear(hidden_size, hidden_size, bias = False))
        
        self.linear_layers.append(nn.Linear(d, 1, bias = True))
        self.convex_layers.append(nn.Linear(hidden_size, 1, bias = False))
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        if positive:
            for convex_layer in self.convex_layers:
                for p in convex_layer.parameters():
                    p.data.clamp_(0)
    
    def forward(self, x):
        y = self.linear_layers[0](x)
        y = self.leaky_relu(y)**2
        for linear_layer, convex_layer in zip(self.linear_layers[1:-1], self.convex_layers[:-1]):
            y = self.leaky_relu(linear_layer(x) + convex_layer(y))
        y = self.linear_layers[-1](x) + self.convex_layers[-1](y)
        return self.leaky_relu(y)


class ICNN_fg:
    def __init__(self, dataset, hidden_size = 64, num_layers = 4, save_folder = "output", g_positive = False, g_penalty = True):
        self.d = dataset["d"]

        self.f = ICNN(self.d, hidden_size, num_layers).to(device)
        self.g = ICNN(self.d, hidden_size, num_layers, positive = g_positive).to(device)

        self.dataset = dataset
        self.init_data(dataset["n_test"])

        self.save_folder = save_folder
        self.to_save = {}
        self.to_save["test_X"] = self.test_X.cpu().detach().numpy()
        self.to_save["test_Y"] = self.test_Y.cpu().detach().numpy()
        self.to_save["POT_distance"] = self.POT_distance
        self.to_save["ICNN_distance"] = []
        self.to_save["x_to_y"] = []
        self.to_save["y_to_x"] = []

        self.g_positive = g_positive
        self.g_penalty = g_penalty

    def init_data(self, n_test):
        self.test_X, self.test_Y = generate_dataset(self.dataset, n_test)
        self.POT_distance = OT(self.test_X, self.test_Y)[0].cpu().detach().item()

    def train(self, t_iters = 1000, k_iters = 10, eval_freq = 50, plot_freq = 100, n_train = 500):
        t0 = time.time()

        optimizer_f = optim.Adam(self.f.parameters(), betas = (0.5,0.9), lr = 1e-3, maximize = True)
        optimizer_g = optim.Adam(self.g.parameters(), betas = (0.5,0.9), lr = 1e-3)

        test_X = torch.tensor(self.test_X, requires_grad = True, dtype = torch.float32).to(device)
        test_Y = torch.tensor(self.test_Y, requires_grad = True, dtype = torch.float32).to(device)
        for t in range(t_iters):
            for k in range(k_iters):

                X, Y = generate_dataset(self.dataset, n_train)
                X = torch.tensor(X, dtype = torch.float32, requires_grad = True).to(device)
                Y = torch.tensor(Y, dtype = torch.float32, requires_grad = True).to(device)

                g_y = self.g(Y)
                grad_gy = torch.autograd.grad(g_y, Y, torch.ones_like(g_y), create_graph = True)[0]
                f_grad_gy = self.f(grad_gy)

                # Optimization in g
                loss_g = (f_grad_gy - (Y * grad_gy).sum(axis = 1, keepdims = True)).mean()
                if self.g_penalty and not self.g_positive:
                    for W in self.g.convex_layers:
                        for p in W.parameters():
                            loss_g += (torch.relu(-p)**2).sum()
                optimizer_g.zero_grad()
                loss_g.backward(retain_graph = True)
                optimizer_g.step()

                if self.g_positive:
                    for convex_layer in self.g.convex_layers:
                        for p in convex_layer.parameters():
                            p.data.clamp_(0)

            # Optimization in f
            g_y = self.g(Y)
            grad_gy = torch.autograd.grad(g_y, Y, torch.ones_like(g_y), create_graph = True)[0]
            f_grad_gy = self.f(grad_gy)
            f_x = self.f(X)
            loss_f = (f_grad_gy - f_x).mean()

            optimizer_f.zero_grad()
            loss_f.backward()
            optimizer_f.step()

            for convex_layer in self.f.convex_layers:
                for p in convex_layer.parameters():
                    p.data.clamp_(0)

            # Evaluation
            if (t % eval_freq == 0) or (t == t_iters - 1):
                approx_distance = self.OT_distance(test_X, test_Y)
                print(f"Iteration: {t} | POT distance: {self.POT_distance:.3f} | ICNN distance: {approx_distance:.3f} | Time: {time.time() - t0:.3f}")
                self.to_save["ICNN_distance"].append((t, time.time() - t0, approx_distance.cpu().detach().item()))

                # display_transport_plan(test_X.detach().cpu().numpy().T, test_Y.detach().cpu().numpy().T, x_to_y.T, y_to_x.T, t, self.POT_distance, approx_distance)
                # save test_X, test_Y, x_to_y, y_to_x, t, self.POT_distance, approx_distance
                if t % plot_freq == 0:
                    x_to_y, y_to_x = self.OT_xy(test_X, test_Y)
                    self.to_save["x_to_y"].append((t, x_to_y))
                    self.to_save["y_to_x"].append((t, y_to_x))
                    self.save()


    def OT_distance(self, x, y):
        """
        Compute W2 distance
        """
        g_y = self.g(y)
        grad_gy = torch.autograd.grad(g_y, y, torch.ones_like(g_y))[0]
        f_grad_gy = self.f(grad_gy)
        f_x = self.f(x)
        norm_x = (x*x).sum(axis=1)
        norm_x = norm_x.mean()
        norm_y = (y*y).sum(axis=1)
        norm_y = norm_y.mean()

        return 2 * (torch.mean(f_grad_gy - f_x  - (y * grad_gy).sum(axis=1, keepdims=True)) + 0.5*(norm_x + norm_y))
    
    def OT_xy(self, x, y):
        """ Compute optimal mappings from X to Y and Y to X """
        g_y = self.g(y)
        T_yx = torch.autograd.grad(g_y, y, torch.ones_like(g_y))[0]
        
        f_x = self.f(x)
        T_xy = torch.autograd.grad(f_x, x, torch.ones_like(f_x))[0]

        return T_xy.cpu().numpy(), T_yx.cpu().numpy()

    def save(self):
        with open(f"{self.save_folder}/data.pkl", "wb") as f:
            pickle.dump(self.to_save, f)