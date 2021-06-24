import argparse
import math
from pathlib import Path
import sys
import os
import datetime
import traceback

sys.path.append('./taming-transformers')
sys.path.append('./CLIP')

from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from tqdm import tqdm

from genfuncs import do_init, train, synth, emit_filename
from genfuncs import get_model, get_z

import clip

base_size=[240, 135]

orig_args = argparse.Namespace(
    image_prompts=[],
    noise_prompt_seeds=[],
    noise_prompt_weights=[],
    init_weight=0.,
    clip_model='ViT-B/32',
    vqgan_config='vqgan_imagenet_f16_1024.yaml',
    vqgan_checkpoint='vqgan_imagenet_f16_1024.ckpt',
    step_size=0.05,
    cut_pow=1.,
    display_freq=50,
    seed=0,
)    

default_prompt = 'flying in the air on a broomstick over a small town with seabirds following along in the style of studio ghibli | artstation | unreal engine'

def main():
    global z, model

    parser = argparse.ArgumentParser(description="Deep learning grid layout")
    # parser.add_argument('--do-hexgrid', default=False, action='store_true',
    #                     help="shift even rows by half a cell size to make grid a hex grid")
    parser.add_argument('--outfile', default="outputs/%DATE%_%WIDTH%x%HEIGHT%_c%CYCLES%_%SEQ%.png",
                        help="single output file")
    parser.add_argument('--random-seed', default=None, type=int,
                        help='Use a specific random seed (for repeatability)')
    parser.add_argument('--scale', default=1, type=float,
                        help='scale to double size, etc.')
    parser.add_argument('--iterations', default=100, type=int,
                        help='Run for how many iterations')
    parser.add_argument('--cutn', default=64, type=int,
                        help='Number of cuts per iteration')
    parser.add_argument('--init-image', default=None,
                        help='Run for how many iterations')
    parser.add_argument('--prompt', default=default_prompt,
                        help='Run for how many iterations')
    args = parser.parse_args()

    template_dict = {}

    if args.random_seed is not None:
        print("Setting random seed: ", args.random_seed)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        template_dict["SEED"] = args.random_seed
    else:
        template_dict["SEED"] = None

    if args.outfile.lower() == "none":
        args.outfile = None

    if args.scale > 1:
        base_size[0] = int(base_size[0] * args.scale)
        base_size[1] = int(base_size[1] * args.scale)

    prompts = [args.prompt]

    do_init(orig_args, base_size, prompts, args.init_image, args.cutn)

    train_cycles = 0
    try:
        for i in tqdm(range(args.iterations)):
            train(i, orig_args.init_weight, orig_args.display_freq)
            train_cycles = train_cycles + 1
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(e)
        traceback.print_exc()
        pass

    template_dict["CYCLES"] = train_cycles
    template_dict["WIDTH"] = base_size[0]
    template_dict["HEIGHT"] = base_size[1]

    if args.outfile is not None:
        z = get_z();
        model=get_model()
        # print(z)
        # print(z.shape)
        out = synth(model, z)
        emitted_filename = emit_filename(args.outfile, template_dict)
        TF.to_pil_image(out[0].cpu()).save(emitted_filename)
        print(f"{args.iterations} {base_size} -> {emitted_filename}")

if __name__ == '__main__':
    main()
