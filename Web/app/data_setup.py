import os
import shutil
import json
import pandas as pd
from flask import current_app
import logging

logger = logging.getLogger(__name__)

def initialize_json(json_file):
    """Create JSON file if it doesn't exist."""
    if not os.path.exists(json_file):
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, "w") as f:
            json.dump([], f)  # Initialize with an empty list

def generate_unique_id():
    """Generate a 10-digit unique ID based on the current timestamp."""
    import time
    return int(time.time() * 1000) % 10000000000

def get_directory_size(directory):
    """Calculate directory size in MB or GB."""
    total_size = sum(
        os.path.getsize(os.path.join(directory, f)) 
        for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    )
    if total_size < 1e9:  # Less than 1 GB
        return f"{total_size / 1e6:.2f} MB"
    else:
        return f"{total_size / 1e9:.2f} GB"

def update_json_record(json_file, task_type, dataset_name, test_count, test_size, gt_count, gt_size, model_count, active_status):
    """Update JSON file with dataset info."""
    with open(json_file, "r+") as f:
        records = json.load(f)

        # Deactivate old records if dataset is replaced
        for record in records:
            if record["Task Type"] == task_type and record["Dataset Name"] == dataset_name:
                record["Active"] = False

        # Create new record entry
        new_record = {
            "Unique ID": generate_unique_id(),
            "Task Type": task_type,
            "Dataset Name": dataset_name,
            "Path": f"/shared/data/{task_type}/{dataset_name}",
            "Number of Models": model_count,
            "Number of Test Images": test_count,
            "Size of Test Images": test_size,
            "Number of Ground Truth Images": gt_count,
            "Size of Ground Truth Images": gt_size,
            "Active": active_status
        }
        records.append(new_record)
        f.seek(0)
        json.dump(records, f, indent=4)
    logger.info(f"Dataset record updated in {json_file}")

def process_dataset_form(task_type, dataset_name, model_files, test_images, ground_truth_images, replace_existing=False):
    try:
        # Define base directory for storing datasets
        app_data_path = os.getenv('APP_DATA_PATH')
        base_dir = os.path.join(app_data_path, task_type, dataset_name)
        test_dir = os.path.join(base_dir, "Test_images")
        ground_truth_dir = os.path.join(base_dir, "Ground_truth")
        model_dir = os.path.join(base_dir, "model")
        mapping_file = os.path.join(base_dir, "mapping.csv")
        json_file = os.path.join(app_data_path, "data_record.json")

        # Initialize JSON file if it doesn't exist
        initialize_json(json_file)

        # Handle dataset replacement
        if os.path.exists(base_dir):
            if replace_existing:
                logger.info(f"Replacing existing dataset at {base_dir}")
                shutil.rmtree(base_dir)
            else:
                return False, "Dataset already exists. Please confirm replacement."

        # Create necessary directories
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Save model files
        for model_file in model_files:
            model_file.save(os.path.join(model_dir, model_file.filename))
        logger.info(f"Saved {len(model_files)} model file(s) to {model_dir}")

        # Save test and ground truth images and map them
        data, unmapped_images = [], []

        logger.info("Starting to map test images to ground truth images.")
        for test_image in test_images:
            # Extract the test ID using the provided method
            test_id = os.path.splitext(test_image.filename)[0].split('_')[-1]
            test_image_path = os.path.join(test_dir, test_image.filename)
            ground_truth_image = f"{dataset_name}_{test_id}_Segmentation.png"
            ground_truth_image_path = os.path.join(ground_truth_dir, ground_truth_image)

            print("test_id: ",test_id, "test_image_path: ", test_image_path, "ground_truth_image: ", ground_truth_image, "ground_truth_image_path: ",ground_truth_image_path )
            logger.debug(f"Processing test image: {test_image.filename}")
            logger.debug(f"Expected ground truth image name: {ground_truth_image}")

            # Check if the corresponding ground truth image exists
            matching_gt_image = next(
                (img for img in ground_truth_images if img.filename == ground_truth_image), None
            )

            if matching_gt_image:
                # Save test and ground truth images
                test_image.save(test_image_path)
                matching_gt_image.save(ground_truth_image_path)

                logger.info(f"Saved test image to {test_image_path}")
                logger.info(f"Saved matching ground truth image to {ground_truth_image_path}")

                # Map images with forward slashes in paths
                data.append({
                    "#": test_id,
                    "image_name": test_image.filename,
                    "test_images": f"Test_images/{test_image.filename}",
                    "ground_truth": f"Ground_truth/{ground_truth_image}"
                })
            else:
                unmapped_images.append(test_image.filename)
                logger.warning(f"No matching ground truth found for test image {test_image.filename}")

        logger.info(f"Mapping complete. Total mapped images: {len(data)}, Unmapped test images: {len(unmapped_images)}")

        # Additional debug info if there are unmapped images
        if unmapped_images:
            logger.debug(f"List of unmapped images: {unmapped_images}")

        # Save mapping to CSV
        df = pd.DataFrame(data, columns=["#", "image_name", "test_images", "ground_truth"])
        df.to_csv(mapping_file, index=False)
        logger.info(f"Mapping file saved at {mapping_file}")
        logger.debug(f"Mapping file contents: {df.head()}")

        # Calculate directory sizes
        test_size = get_directory_size(test_dir)
        gt_size = get_directory_size(ground_truth_dir)

        # Update JSON with dataset record
        update_json_record(
            json_file=json_file,
            task_type=task_type,
            dataset_name=dataset_name,
            test_count=len(test_images),
            test_size=test_size,
            gt_count=len(data),
            gt_size=gt_size,
            model_count=len(model_files),
            active_status=True
        )

        # Return success message with details about unmapped images, if any
        if unmapped_images:
            return True, f"Dataset processed successfully, but the following images were unmapped: {', '.join(unmapped_images)}"
        return True, "Dataset processed successfully."

    except Exception as e:
        logger.error("An error occurred while processing the dataset form", exc_info=True)
        return False, f"An error occurred: {str(e)}"
