import os
import torch
import geoopt
from utils.general_utils import get_expon_lr_func
from utils.system_utils import searchForMaxIteration
from utils.tilted_utils import So3

class TiltedModel:
    def __init__(self, num_points):
        self.tilted = So3(num_points).cuda()
        self.optimizer = None

    def step(self, xyz, iters):
        return self.tilted(xyz, iters)

    # Reference to https://github.com/geoopt/geoopt, "Riemannian Adaptive Optimization Methods" ICLR 2019
    def train_setting(self, training_args):
         self.optimizer = geoopt.optim.RiemannianAdam([{"params": [self.tilted.tau], 'lr': training_args.quater_lr_init, 'name': 'tau'}])
         self.tilted_scheduler_args = get_expon_lr_func(lr_init=training_args.quater_lr_init,
                                                        lr_final=0.0,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.quater_lr_max_step)
    
    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "tilted/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.tilted.state_dict(), os.path.join(out_weights_path, 'tilted.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "tilted"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "tilted/iteration_{}/tilted.pth".format(loaded_iter))
        self.tilted.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "tilted":
                lr = self.tilted_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr