import argparse
import os
import nibabel as nib
import sys
import random
import threading
import itertools
import time
sys.path.append(".")
import numpy as np
import torch as th
from PIL import Image
from guided_diffusion import logger
from guided_diffusion.isicloader import ISICDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
from datetime import timedelta

# Set seeds for reproducibility
seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Spinner function to display progress while the process is running
def spinner():
    for symbol in itertools.cycle(['⠋', '⠙', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇']):
        sys.stdout.write(f'\r{symbol} Processing... ')
        sys.stdout.flush()
        time.sleep(0.1)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def main():
    # Parse arguments
    args = create_argparser().parse_args()

    # Set the device to use (GPU if available, otherwise CPU)
    device = th.device(f"cuda:{args.gpu_dev}" if th.cuda.is_available() else "cpu")
    logger.configure(dir=args.out_dir)

    # Start the spinner in a separate thread
    spinner_thread = threading.Thread(target=spinner)
    spinner_thread.daemon = True
    spinner_thread.start()

    # Prepare the data loader (using ISIC dataset)
    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test, mode='Test')
        args.in_ch = 4
    
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    # Create model and diffusion processes
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Load the model's state dictionary
    state_dict = th.load(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v  # Remove 'module.' from keys if it's there
        else:
            new_state_dict = state_dict
    model.load_state_dict(new_state_dict)

    # Move the model to the appropriate device (GPU or CPU)
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()

    # Get the number of samples
    num_samples = len(data)
    print(f"Number of samples: {num_samples}")

    # Print number of loops (i.e., iterations through the dataset)
    print(f"Number of iterations (for loop runs): {num_samples}")

    # Variables for time estimation
    total_time = 0
    start_time = time.time()

    # Loop over the dataset and generate samples
    for i in range(num_samples):
        loop_start_time = time.time()  # Start time for this iteration

        b, m, path = next(data)  # Get batch from data loader
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)  # Add noise channel

        # Get slice ID for naming outputs
        slice_ID = path[0].split("_")[-1].split('.')[0] if args.data_name == 'ISIC' else path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        # Generate ensemble of segmentation masks
        for j in range(args.num_ensemble):
            model_kwargs = {}
            start.record()
            
            # If dpm_solver is part of your diffusion model, use it here
            if hasattr(diffusion, 'dpm_solver_sample_loop'):
                sample_fn = diffusion.dpm_solver_sample_loop
            else:
                # Default to p_sample_loop_known or ddim_sample_loop_known
                sample_fn = diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            
            # Run the sampling function
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                img.to(device),
                step=args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('Time for 1 sample', start.elapsed_time(end))  # Time measurement

            co = th.tensor(cal_out)
            if args.version == 'new':
                enslist.append(sample[:, -1, :, :])
            else:
                enslist.append(co)

            # Optional debug visualization
            if args.debug:
                if args.data_name == 'ISIC':
                    o = th.tensor(org)[:, :-1, :, :]
                    c = th.tensor(cal).repeat(1, 3, 1, 1)
                    s = sample[:, -1, :, :]
                    b, h, w = s.size()
                    ss = s.clone()
                    ss = ss.view(s.size(0), -1)
                    ss -= ss.min(1, keepdim=True)[0]
                    ss /= ss.max(1, keepdim=True)[0]
                    ss = ss.view(b, h, w)
                    ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)
                    tup = (ss, o, c)
                compose = th.cat(tup, 0)
                vutils.save_image(compose, fp=os.path.join(args.out_dir, f'{slice_ID}_output{j}.jpg'), nrow=1, padding=10)

        # Perform ensemble averaging with STAPLE
        ensres = staple(th.stack(enslist, dim=0)).squeeze(0)
        vutils.save_image(ensres, fp=os.path.join(args.out_dir, f'{slice_ID}_output_ens.jpg'), nrow=1, padding=10)

        # Calculate time taken for this iteration
        loop_time = time.time() - loop_start_time
        total_time += loop_time
        avg_time_per_loop = total_time / (i + 1)
        remaining_time = avg_time_per_loop * (num_samples - (i + 1))

        # Print estimated remaining time
        print(f"Iteration {i + 1}/{num_samples} completed. Estimated remaining time: {timedelta(seconds=int(remaining_time))}")

    # Stop the spinner once the process is complete
    sys.stdout.write('\r✔ Process Completed!\n')
    sys.stdout.flush()

def create_argparser():
    # Default arguments
    defaults = dict(
        data_name='ISIC',
        data_dir="../MedSegDiff/data/ISIC/Test",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",  # Path to pre-trained model
        num_ensemble=5,  # Number of samples in the ensemble
        gpu_dev="0",
        out_dir='./results/',
        debug=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
