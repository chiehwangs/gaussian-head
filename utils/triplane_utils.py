import torch
import itertools
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

relu = torch.nn.ReLU()

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, feature_dim , trans_num: int, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)  

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))  
    
    interp_f = 1.
    for i in range(trans_num):
        interp_grid = grid[:, i*feature_dim:(i+1)*feature_dim, :, :]
        B, feature_dim = interp_grid.shape[:2]
        n = coords.shape[-2]
        interp = grid_sampler(
            interp_grid,  
            coords, 
            align_corners=align_corners,
            mode='bilinear', padding_mode='border')  
        
        interp_f *= interp  

    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  
    interp = interp.squeeze()  
    return interp

def init_planes(grid_dim, in_dim, out_dim, resolution, a, b):
    planes = list(itertools.combinations(range(in_dim), grid_dim))
    plane_coefs = nn.ParameterList()
    for i, plane in enumerate(planes):
        init_plane_coef = nn.Parameter(torch.empty([1, out_dim] + [resolution[cc] for cc in plane[::-1]]))  
        nn.init.uniform_(init_plane_coef, a=a, b=b)
        plane_coefs.append(init_plane_coef)
    return plane_coefs

def interpolate_ms_features(points, triplane, trans_num, plane_dim, concat_f, num_levels):
    planes = list(itertools.combinations(range(points.shape[-1]), plane_dim)) 
    multi_scale_interp = [] if concat_f else 0.
    plane : nn.ParameterList
    for scale, plane in enumerate(triplane[:num_levels]):  
        interp_space = 1.
        for ci , coo_comb in enumerate(planes): 
            feature_dim = (plane[ci].shape[1]) // trans_num  
            interp_out_plane = (grid_sample_wrapper(plane[ci], points[..., coo_comb], feature_dim, trans_num).view(-1, feature_dim))  
            # product reduce
            interp_space = interp_space * interp_out_plane

        if concat_f:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_f:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp

class MLP(nn.Module):
    def __init__(self, mlptype, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.mlptype = mlptype

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                if self.mlptype == 'opacity':
                    x = torch.sigmoid(x)  
                elif self.mlptype == 'color':
                    x = relu(x)
        if self.mlptype == 'opacity':
            return x[..., :1], x[..., 1:]  
        else:
            return x

class TriPlaneNetwork(nn.Module):
    def __init__(self, tilted, warm_up, sh_degree, grid_dim=2, in_dim=3, out_dim=32, resolution=[64,64,64], a=0.1, b=0.5):
        super().__init__()
        self.grid_dim = grid_dim 
        self.in_dim = in_dim  
        self.out_dim = out_dim  
        self.base_resolution = resolution  
        self.multi_scale_res = [1,2,4,8]
        self.concat_feature = False
        self.warm_up = warm_up

        assert self.in_dim == len(self.base_resolution), "Resolution must have same number of elements as input-dimension"
        self.tri_plane = nn.ModuleList()
        for res_scale in self.multi_scale_res:
            resolution = [res_scale * resolution for resolution in self.base_resolution]
            plane_coefs = init_planes(self.grid_dim, self.in_dim, self.out_dim, resolution, a=a, b=b)
            if self.concat_feature:
                self.feature_dim += plane_coefs[-1].shape[1] 
            else:
                self.feature_dim = plane_coefs[-1].shape[1]
            self.tri_plane.append(plane_coefs)  
        
        self.geo_feat_dim = out_dim
        self.hidden_dim = 64
        self.num_layers = 3
        self.opacity_net = MLP(mlptype='opacity', dim_in=self.feature_dim, dim_out=1 + self.geo_feat_dim,
                               dim_hidden=self.hidden_dim, num_layers=self.num_layers) 
        self.view_dim = 3
        self.hidden_dim_view = 64
        self.num_layers_view = 2
        self.shs_coefs = (sh_degree + 1) ** 2
        self.shs_out_dim = 3*((self.shs_coefs)-1) + 3
        self.shs_net = MLP(mlptype='shs', dim_in=self.view_dim + self.geo_feat_dim, dim_out=self.shs_out_dim,
                           dim_hidden=self.hidden_dim_view, num_layers=self.num_layers_view)
        self.shs_dc_net = nn.Linear(self.shs_out_dim, 3)
        self.shs_rest_net = nn.Linear(self.shs_out_dim, 45)

        self.tilted_model = tilted
        
    def forward(self, xyz, dirs, iters,):  
        assert len(xyz.shape) == 2 and xyz.shape[-1] == self.in_dim, 'input points dim must be (num_points, 3)'

        interp_coords = self.tilted_model.step(xyz, iters)  
        canonical_f = []
        for p in range(interp_coords.shape[0]):
            interp_coord = interp_coords[p]  
            interp_ms_feat = interpolate_ms_features(interp_coord, self.tri_plane, trans_num=interp_coords.shape[0], 
                                                    plane_dim=self.grid_dim, concat_f=self.concat_feature, num_levels=None)  
            canonical_f.append(interp_ms_feat)
        canonical_f = torch.cat(canonical_f, dim=1)
        # else:
        #     canonical_f = interpolate_ms_features(xyz, self.tri_plane,  trans_num=1, plane_dim=self.grid_dim, 
                                                    #  concat_f=self.concat_feature, num_levels=None)
        
        opacity, geo_feat = self.opacity_net(canonical_f)
        opacity = torch.sigmoid(opacity)
        in_shs_net = torch.cat([geo_feat, dirs], dim=-1)

        features_sh = self.shs_net(in_shs_net)  # only geo_feat, no dirs will causing a sharp drop in performance
        features_dc = self.shs_dc_net(features_sh)
        features_rest = self.shs_rest_net(features_sh)

        return opacity, features_dc, features_rest

        
