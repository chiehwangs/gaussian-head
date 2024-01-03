import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel

from scene.tri_plane import TriPlaneModel
from utils.sh_utils import eval_sh


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

def component_norms(a):
    return torch.sum(a**2, dim=("g", "transform_count"), keepdim=False)

def render(viewpoint_camera, pc: GaussianModel, triplane: TriPlaneModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, iteration,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points

     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:  # conpute convariance 3D in advance rather in rasterization
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation  

    shs = None
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_xyz.shape[0], 1))  
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    opacity, feature_dc, feature_rest = triplane.step(xyz=means3D, dirs=dir_pp_normalized, iteration=iteration)  
    shs = torch.cat([feature_dc.view(-1, 1, 3), feature_rest.view(-1, 15, 3)], dim=1)

    pc._features_dc, pc._features_rest, pc._opacity = shs[:, :1, :], shs[:, 1:, :], opacity
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)  
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))  
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)  
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features  
    else:
        colors_precomp = override_color

    rendered_image, radii, depth = rasterizer(
        means3D=means3D,  
        means2D=means2D,  
        shs=shs,  
        colors_precomp=colors_precomp,  
        opacities=opacity, 
        scales=scales, 
        rotations=rotations, 
        cov3D_precomp=cov3D_precomp) 

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth}
