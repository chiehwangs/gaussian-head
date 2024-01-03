import torch
from scene import Scene
from scene.deform_model import DeformModel
from scene.tilted_model import TiltedModel
from scene.tri_plane import TriPlaneModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


def render_set(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, triplane, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpt_on_the_fly:
            view.load2device()
        exp = view.exp
        xyz = gaussians.get_xyz
        exp_input = exp.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), exp_input)
        results = render(view, gaussians, triplane, pipeline, background, d_xyz, d_rotation, d_scaling, iteration)
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, warm_up: int, is_debug: bool, skip_train: bool, skip_test: bool,
                mode: str, novel_view, only_head):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, is_debug=is_debug, novel_view=novel_view, only_head=only_head, load_iteration=iteration, shuffle=False)
        exp_dims = scene.train_cameras[1.0][0].exp.size()
        deform = DeformModel(exp_dims[0])
        deform.load_weights(dataset.model_path)
        
        tilted = TiltedModel(gaussians.get_xyz.shape[0])
        tilted.load_weights(dataset.model_path)

        triplane = TriPlaneModel(warm_up, dataset.sh_degree, tilted=tilted)
        triplane.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, "train",  scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, triplane, pipeline,
                        background, deform,)
            
        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, "test", scene.loaded_iter, 
                        scene.getTestCameras(), gaussians, triplane, pipeline,
                        background, deform,)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render'])
    parser.add_argument("--is_debug", type=bool, default=False)
    parser.add_argument("--warm_up", type=int, default=3_000)
    parser.add_argument("--novel_view", type=bool, default=False)
    parser.add_argument("--only_head", type=bool, default=False)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.warm_up, args.is_debug, args.skip_train, args.skip_test, 
                args.mode, args.novel_view, args.only_head)
