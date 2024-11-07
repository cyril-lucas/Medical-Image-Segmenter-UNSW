import os
import shutil
import json
import pandas as pd
from datetime import datetime
import time
import logging
from werkzeug.utils import secure_filename


logger = logging.getLogger(__name__)

def initialize_data_record(json_file_path):
    """Initialize the JSON file if it doesn't exist."""
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    if not os.path.exists(json_file_path):
        with open(json_file_path, "w") as f:
            json.dump([], f)
        logger.info(f"Initialized data record at {json_file_path}")

def generate_unique_id(json_file):
    """Generate a unique 10-digit ID that does not exist in the JSON file."""
    existing_ids = set()
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
            existing_ids = {record["Unique ID"] for record in data}

    while True:
        unique_id = int(time.time() * 1000) % 10000000000
        if unique_id not in existing_ids:
            logger.info(f"Generated unique ID: {unique_id}")
            return unique_id

def save_files(files, save_dir, file_type, dataset_name=None, is_ground_truth=False):
    """
    Save uploaded files to a specified directory with appropriate naming conventions.
    Logs a simplified message after saving.
    
    Args:
        files: List of FileStorage objects to save.
        save_dir: Directory to save files.
        file_type: Type of files being saved (e.g., "Test_images", "Ground_truth").
        dataset_name: Name of the dataset for prefixing (e.g., "ISIC").
        is_ground_truth: Boolean indicating if the files are ground truth images.
    
    Returns:
        List of saved filenames with appropriate naming conventions.
    """
    paths = []
    os.makedirs(save_dir, exist_ok=True)
    simple_filenames = []

    for file in files:
        # Secure the filename
        original_filename = secure_filename(file.filename)
        
        # Extract the ID part of the filename
        id_part = original_filename.split('_')[-1].split('.')[0]  # e.g., "0012169"
        
        # Construct the new filename
        if is_ground_truth:
            new_filename = f"{dataset_name}_{id_part}_Segmentation.png"
        else:
            new_filename = f"{dataset_name}_{id_part}.jpg"
        
        # Save the file
        file_path = os.path.join(save_dir, new_filename)
        file.save(file_path)
        paths.append(new_filename)
        simple_filenames.append(new_filename)

    # Log a sample of filenames saved
    logger.info(f"{', '.join(simple_filenames[:3])}... stored in {file_type}")

    return paths

def deactivate_existing_dataset(task_type, dataset_name, json_file):
    """Deactivate old records if the dataset is being replaced."""
    if not os.path.exists(json_file):
        return

    with open(json_file, "r+") as f:
        records = json.load(f)
        for record in records:
            if record["Task Type"] == task_type and record["Dataset Name"] == dataset_name:
                record["Active"] = False
        f.seek(0)
        f.truncate()
        json.dump(records, f, indent=4)

def process_data_setup_form(task_type, dataset_name, model_files, test_images, ground_truth_images, json_file):
    """Process form data, validate inputs, save files, and record dataset metadata."""
    logger.info(f"Starting data setup for dataset: '{dataset_name}' under task: '{task_type}'")
    initialize_data_record(json_file)

    # Base directories
    base_data_path = os.getenv("APP_DATA_PATH", "../shared/data")
    base_dir = os.path.join(base_data_path, task_type, dataset_name)
    test_dir = os.path.join(base_dir, "Test_images")
    ground_truth_dir = os.path.join(base_dir, "Ground_truth")
    model_dir = os.path.join(base_dir, "model")
    mapping_file = os.path.join(base_dir, "mapping.csv")

    # Check if dataset already exists
    if os.path.exists(base_dir):
        logger.warning(f"Dataset '{dataset_name}' already exists. Deactivating and replacing...")
        deactivate_existing_dataset(task_type, dataset_name, json_file)
        shutil.rmtree(base_dir)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save model files
    model_file_paths = save_files(model_files, model_dir, "Model files")
    logger.info(f"Model files saved: {', '.join(model_file_paths[:3])}...")

    # Save test and ground truth images with appropriate naming conventions
    test_image_names = save_files(test_images, test_dir, "Test_images", dataset_name=dataset_name)
    ground_truth_names = save_files(ground_truth_images, ground_truth_dir, "Ground_truth", dataset_name=dataset_name, is_ground_truth=True)
    logger.info(f"Test images saved in 'Test_images': {', '.join(test_image_names[:3])}...")
    logger.info(f"Ground truth images saved in 'Ground_truth': {', '.join(ground_truth_names[:3])}...")

    # Check for matching counts of test and ground truth images
    if len(test_image_names) != len(ground_truth_names):
        error_msg = "The number of test images and ground truth images do not match."
        logger.error(error_msg)
        return f"Error: {error_msg}"

    # Map test images to ground truth images using only the ID part
    mapping_data = []
    unmapped_images = []

    for test_image in test_image_names:
        test_id = os.path.splitext(test_image)[0].split('_')[-1]  # Extract ID part
        gt_image_name = f"{dataset_name}_{test_id}_Segmentation.png"

        if gt_image_name in ground_truth_names:
            mapping_data.append({
                "#": test_id,
                "image_name": test_image,
                "test_images": f"Test_images/{test_image}",
                "ground_truth": f"Ground_truth/{gt_image_name}"
            })
        else:
            unmapped_images.append(test_image)
            logger.warning(f"Unmapped test image detected: {test_image}")

    # Save the mapping to CSV
    if mapping_data:
        pd.DataFrame(mapping_data).to_csv(mapping_file, index=False)
        logger.info(f"Mapping file saved as 'mapping.csv' in dataset directory")

    # Update JSON record if there are no unmapped images
    if not unmapped_images:
        test_size = get_directory_size(test_dir)
        gt_size = get_directory_size(ground_truth_dir)
        record = {
            "Unique ID": generate_unique_id(json_file),
            "Task Type": task_type,
            "Dataset Name": dataset_name,
            "Path": base_dir,
            "Number of Models": len(model_file_paths),
            "Number of Test Images": len(test_image_names),
            "Size of Test Images": test_size,
            "Number of Ground Truth Images": len(ground_truth_names),
            "Size of Ground Truth Images": gt_size,
            "Timestamp": datetime.now().isoformat(),
            "Active": True
        }
        append_to_data_record(record, json_file)
        logger.info(f"Dataset '{dataset_name}' processed successfully and added to records.")
        return f"Dataset '{dataset_name}' processed successfully."
    else:
        error_msg = f"Unmapped images detected: {', '.join(unmapped_images)}"
        logger.error(error_msg)
        return error_msg

def append_to_data_record(record, json_file):
    """Append a new record to the data_record.json file."""
    with open(json_file, "r+") as f:
        data = json.load(f)
        data.append(record)
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=4)
    logger.info("Data record updated with new dataset information.")

def get_directory_size(directory):
    """Calculate the size of a directory in MB or GB."""
    total_size = sum(
        os.path.getsize(os.path.join(directory, f)) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    )
    size_str = f"{total_size / 1e6:.2f} MB" if total_size < 1e9 else f"{total_size / 1e9:.2f} GB"
    logger.info(f"Directory size for {directory}: {size_str}")
    return size_str