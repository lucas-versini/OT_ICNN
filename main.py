from models import ICNN_fg
import os

dataset_parameter = {"type": "gaussians",
                    "n_train": 200, "n_test": 1000,
                    "d": 100}


dataset_parameter = {"type": "two_circles",
                    "n_train": 200, "n_test": 1000,
                    "d": 2}

dataset_parameter = {"type": "Voronoi",
                    "n_train": 1000, "n_test": 1000,
                    "d": 2}

save_folder = "output"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
icnn_fg = ICNN_fg(dataset_parameter, num_layers = 5, hidden_size = 128, save_folder = save_folder)

icnn_fg.train(k_iters = 10, t_iters = 10000, eval_freq = 10, plot_freq = 100)
