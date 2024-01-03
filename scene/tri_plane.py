import os
import torch
from utils.system_utils import searchForMaxIteration
from utils.triplane_utils import TriPlaneNetwork
from utils.general_utils import get_expon_lr_func

class TriPlaneModel:
    def __init__(self, warm_up, sh_degree, tilted):
        self.triplane = TriPlaneNetwork(tilted, warm_up, sh_degree).cuda()
        self.optimizer = None
    
    def step(self, xyz, dirs, iteration):
        return self.triplane(xyz, dirs, iteration)

    def train_setting(self, training_args):
        parameter = []
        for module in self.triplane.tri_plane:
            parameter.extend(module.parameters())
        l = [{"params": parameter, 'lr': training_args.triplane_lr, 'name':"trip"},
             {"params": list(self.triplane.opacity_net.parameters()), 'lr': training_args.opacity_lr, 'name': 'trip.opacity'},
             {"params": list(self.triplane.shs_net.parameters()), 'lr': training_args.feature_lr, 'name': 'trip.shs'},
             {'params': list(self.triplane.shs_dc_net.parameters()), 'lr': training_args.feature_lr, 'name': 'trip.shs_dc'},
             {'params': list(self.triplane.shs_rest_net.parameters()), 'lr': training_args.feature_lr / 20.0, 'name': 'trip.shs_rest'}]
       
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.triplane_scheduler_args = get_expon_lr_func(lr_init=training_args.triplane_lr, 
                                                         lr_final=training_args.position_lr_final,
                                                         lr_delay_mult=training_args.position_lr_delay_mult,
                                                         max_steps=training_args.triplane_lr_max_step)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "trip/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.triplane.state_dict(), os.path.join(out_weights_path, 'trip.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "trip"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "trip/iteration_{}/trip.pth".format(loaded_iter))
        self.triplane.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "trip":
                lr = self.triplane_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def print_model_parameters(self):
        Model_parameters = 0
        for name, module in self.triplane.named_children():
            total_params = sum(p.numel() for p in module.parameters())
            print(f"{name} parameters: {total_params}")
            Model_parameters += total_params
        print(f'Total parameters: {Model_parameters}')