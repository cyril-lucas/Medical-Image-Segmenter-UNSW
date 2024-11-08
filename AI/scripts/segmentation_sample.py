import os
import sys
import json
import random
import io
import argparse
import numpy as np
import torch as th
import threading
import itertools
import time
from datetime import timedelta
from collections import OrderedDict
from guided_diffusion import dist_util
from guided_diffusion.isicloader import ISICDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision import transforms
from torchsummary import summary  # For model summary

# Set up the logger used in app.py
import logging
log_dir = os.getenv("APP_LOG_PATH", "/shared/logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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

def load_json_arguments(file_path):
    # Load default arguments from arguments.json
    with open(file_path, "r") as f:
        return json.load(f)
    
def filter_state_dict(model, state_dict):
    # Filter out unmatched layers and log them
    filtered_state_dict = {}
    model_state_dict = model.state_dict()

    for k, v in state_dict.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            filtered_state_dict[k] = v
        else:
            logger.warning(f"Skipping layer '{k}' due to shape mismatch: {v.shape} vs {model_state_dict.get(k, 'Not found')}")

    return filtered_state_dict

def save_model_summary(model, input_size, file_path="summary.txt"):
    """
    Save the model summary to a file.
    """
    try:
        # Redirect stdout to capture the summary output
        with io.StringIO() as buffer, open(file_path, "w") as f:
            sys.stdout = buffer  # Redirect stdout to buffer
            summary(model, input_size)  # Generate the summary
            sys.stdout = sys.__stdout__  # Reset stdout
            f.write(buffer.getvalue())  # Write buffer content to file
        logger.info(f"Model summary saved to {file_path}")
    except Exception as e:
        sys.stdout = sys.__stdout__  # Ensure stdout is reset on error
        logger.error(f"Error generating model summary: {e}")
    
def main():

    # Parse specific arguments passed from app.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True, help="Dataset name")
    parser.add_argument("--data_dir", required=True, help="Directory for input data")
    parser.add_argument("--out_dir", required=True, help="Directory for output data")
    parser.add_argument("--model_path", required=True, help="Path to the model")

    # Load and merge defaults from model_and_diffusion_defaults
    defaults = model_and_diffusion_defaults()
    add_dict_to_argparser(parser, defaults)
    cmd_args = parser.parse_args()

    # Load additional arguments from arguments.json
    json_args = load_json_arguments("scripts/arguments.json")
    json_args.update(vars(cmd_args))  # Override with command-line args if provided
    args = argparse.Namespace(**json_args)

    logger.info(f"--data_name: {args.data_name}")
    logger.info(f"--data_dir: {args.data_dir}")
    logger.info(f"--out_dir: {args.out_dir}")
    logger.info(f"--model_path: {args.model_path}")
    
    seed = 10
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    dist_util.setup_dist(args)
    logger.info(f"Output directory is set to: {args.out_dir}")

    # Start the spinner in a separate thread
    spinner_thread = threading.Thread(target=spinner)
    spinner_thread.daemon = True
    spinner_thread.start()

    logger.info("Setting up data transformations...")
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    logger.info("Loading dataset...")
    ds = ISICDataset(args, args.data_dir, transform_test)
    args.in_ch = 4

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.info("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Save the model summary to summary.txt
    input_size = (args.in_ch, args.image_size, args.image_size)
    save_model_summary(model, input_size)

    logger.info("Loading model checkpoint...")
    # Load the state dictionary directly without modifying the keys
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")

    # Filter state_dict to match model's architecture
    filtered_state_dict = filter_state_dict(model, state_dict)
    model.load_state_dict(filtered_state_dict, strict=False)

    model.to('cpu')
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.info("Starting sampling process...")
    num_samples = len(data)
    logger.info(f"Number of samples to process: {num_samples}")

    # Variables for time estimation
    total_time = 0

    for i, (b, m, path) in enumerate(data):
        loop_start_time = time.time()
        slice_ID = path[0].split("_")[-1].split('.')[0] if args.data_name == 'ISIC' else f"sample_{i}"
        logger.info(f"Processing sample {i + 1}/{num_samples}, ID: {slice_ID}")

        c = th.randn_like(b[:, :1, ...])  # Add noise channel
        img = th.cat((b, c), dim=1)
        enslist = []

        for j in range(args.num_ensemble):
            model_kwargs = {}
            sample_fn = diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            
            start_event = th.cuda.Event(enable_timing=True)
            end_event = th.cuda.Event(enable_timing=True)
            start_event.record()

            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                img,
                step=args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            end_event.record()
            th.cuda.synchronize()
            logger.info(f"Sample generation time for ensemble {j + 1}: {start_event.elapsed_time(end_event)} ms")

            enslist.append(sample[:, -1, :, :] if args.version == 'new' else cal_out.clone().detach())

        # Create and save final ensemble result
        ensres = staple(th.stack(enslist, dim=0)).squeeze(0)
        ensres_normalized = visualize(ensres)
        output_path = os.path.join(args.out_dir, f"{slice_ID}_output_ens.jpg")
        vutils.save_image(ensres_normalized, fp=output_path)
        logger.info(f"Saved ensemble output image at: {output_path}")

        # Time tracking for this iteration
        loop_time = time.time() - loop_start_time
        total_time += loop_time
        avg_time_per_loop = total_time / (i + 1)
        remaining_time = avg_time_per_loop * (num_samples - (i + 1))
        logger.info(f"Iteration {i + 1}/{num_samples} completed. Estimated remaining time: {timedelta(seconds=int(remaining_time))}")

    # Stop the spinner once the process is complete
    sys.stdout.write('\r✔ Process Completed!\n')
    sys.stdout.flush()


if __name__ == "__main__":
    main()
