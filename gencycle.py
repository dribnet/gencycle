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

import clip

def real_glob(rglob):
    glob_list = braceexpand(rglob)
    files = []
    for g in glob_list:
        files = files + glob.glob(g)
    return sorted(files)

# this function can fill in placeholders for %DATE%, %SIZE% and %SEQ%
def emit_filename(filename, template_dict):
    datestr = datetime.datetime.now().strftime("%Y%m%d")
    filename = filename.replace('%DATE%', datestr)

    for key in template_dict:
        pattern = "%{}%".format(key)
        value = "{}".format(template_dict[key])
        filename = filename.replace(pattern, value)

    if '%SEQ%' in filename:
        # determine what the next available number is
        cur_seq = 1
        candidate = filename.replace('%SEQ%', "{:02d}".format(cur_seq))
        while os.path.exists(candidate):
            cur_seq = cur_seq + 1
            candidate = filename.replace('%SEQ%', "{:02d}".format(cur_seq))
        filename = candidate
    return filename

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        # print("F", sideY, sideX, min_size, max_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            resampled = resample(cutout, (self.cut_size, self.cut_size))
            # if _ == 0:
            #     print("X", self.cut_pow, torch.rand([]), size, offsetx, offsety)
            #     print(cutout.shape, resampled.shape)
            #     save_image(cutout, 'outputs/cutout5.png')
            #     save_image(resampled, 'outputs/resampled5.png')
            cutouts.append(resampled)
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


class MakeWideCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        box_size = max(sideX, sideY)
        # too lazy to consider non-landscape mode
        max_down_offset = box_size - sideY
        cutouts = []
        for _ in range(self.cutn):
            down_offset = int(torch.rand([]) * max_down_offset)
            fill_color = torch.rand([])
            big_box = torch.cuda.FloatTensor(1, 3, box_size, box_size).fill_(fill_color)
            big_box[:, :, down_offset:(sideY+down_offset), 0:sideX] = input
            resampled = resample(big_box, (self.cut_size, self.cut_size))
            # if _ == 0:
            #     print(big_box.shape, resampled.shape)
            #     save_image(big_box, 'outputs/big_box0.png')
            #     save_image(resampled, 'outputs/resampled0.png')
            cutouts.append(resampled)
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)

class MakeFineCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        do_resample = False
        size = self.cut_size
        min_side = min(sideX, sideY)
        if min_side < size:
            # guess we have to scale UP
            size = min_side
            do_resample = True
            # print("SCALING UP")
        cutouts = []
        for _ in range(self.cutn):
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            if do_resample:
                resampled = resample(cutout, (self.cut_size, self.cut_size))    
            else:
                resampled = cutout
            # if _ == 0:
            # #     print("X", self.cut_pow, torch.rand([]), size, offsetx, offsety)
            #     print(cutout.shape, resampled.shape)
            #     save_image(cutout, 'outputs/cutout_n.png')
            #     save_image(resampled, 'outputs/resampled_n.png')
            cutouts.append(resampled)
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)

class MakeAllCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        # wide_cutn = int(3 * cutn / 4)
        # normal_cutn = 1;
        # fine_cutn = int(1 * cutn / 4)
        # wide_cutn = int(1 * cutn / 3)
        # normal_cutn = int(1 * cutn / 3)
        # fine_cutn = int(1 * cutn / 3)
        wide_cutn = int(1 * cutn / 2)
        normal_cutn = int(1 * cutn / 4)
        fine_cutn = int(1 * cutn / 4)
        self.wide_cutouts = MakeWideCutouts(cut_size, wide_cutn, cut_pow)
        self.cutouts = MakeCutouts(cut_size, normal_cutn, cut_pow)
        self.fine_cutouts = MakeFineCutouts(cut_size, fine_cutn, cut_pow)

    def forward(self, input):
        c1 = self.cutouts(input)
        c2 = self.wide_cutouts(input)
        c3 = self.fine_cutouts(input)
        c4 = torch.cat((c1, c2, c3), 0)
        # print(c1.shape, c2.shape, c3.shape)
        return c3

def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, InterpolationMode.LANCZOS)

def synth(model, z):
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

z = None
opt = None
model = None
perceptor = None
normalize = None
make_cutouts = None
pMs = None
z_min = None
z_max = None

@torch.no_grad()
def checkin(i, losses):
    global z, model
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
    out = synth(model, z)
    TF.to_pil_image(out[0].cpu()).save('progress.png')
    # display.display(display.Image('progress.png'))

def ascend_txt(init_weight):
    global z, model, perceptor, normalize, make_cutouts
    global pMs
    out = synth(model, z)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

    result = []

    if init_weight:
        result.append(F.mse_loss(z, z_orig) * init_weight / 2)

    for prompt in pMs:
        result.append(prompt(iii))

    return result

def train(i, init_weight, display_freq):
    global z, opt
    global z_min, z_max
    opt.zero_grad()
    lossAll = ascend_txt(init_weight)
    if i % display_freq == 0:
        checkin(i, lossAll)
    loss = sum(lossAll)
    loss.backward()
    opt.step()
    with torch.no_grad():
        z.copy_(z.maximum(z_min).minimum(z_max))

def do_init(args, im_size, prompts, init_image, cutn):
    global z, opt
    global model, perceptor, normalize, make_cutouts
    global pMs, z_min, z_max

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
    perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution
    # print(cut_size) # 224! thats tiny
    # sys.exit(0)
    # cut_size = cut_size * 2
    e_dim = model.quantize.e_dim
    f = 2**(model.decoder.num_resolutions - 1)
    make_cutouts = MakeAllCutouts(cut_size, cutn, cut_pow=args.cut_pow)
    n_toks = model.quantize.n_e
    toksX, toksY = im_size[0] // f, im_size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if init_image:
        pil_image = Image.open(init_image).convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args.step_size)

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []

    for prompt in prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

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
        out = synth(model, z)
        emitted_filename = emit_filename(args.outfile, template_dict)
        TF.to_pil_image(out[0].cpu()).save(emitted_filename)
        print(f"{args.iterations} {base_size} -> {emitted_filename}")

if __name__ == '__main__':
    main()
