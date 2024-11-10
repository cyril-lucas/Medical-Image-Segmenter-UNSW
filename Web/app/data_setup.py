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

def process_dataset_form(task_type, dataset_name, model_files, test_images, ground_truth_images):
    try:
        # Define base directory for storing datasets
        app_data_path = current_app.config['APP_DATA_PATH']
        base_dir = os.path.join(app_data_path, task_type, dataset_name)
        test_dir = os.path.join(base_dir, "Test_images")
        ground_truth_dir = os.path.join(base_dir, "Ground_truth")
        model_dir = os.path.join(base_dir, "model")
        mapping_file = os.path.join(base_dir, "mapping.csv")
        json_file = os.path.join(app_data_path, "data_record.json")

        # Initialize JSON file if it doesn't exist
        initialize_json(json_file)

        # Handle dataset replacement if it exists
        if os.path.exists(base_dir):
            logger.info(f"Replacing existing dataset at {base_dir}")
            shutil.rmtree(base_dir)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Save model files
        for model_file in model_files:
            model_file.save(os.path.join(model_dir, model_file.filename))
        logger.info(f"Saved {len(model_files)} model file(s) to {model_dir}")

        # Save test and ground truth images and map them
        data = []
        unmapped_images = []
        for test_image in test_images:
            test_id = test_image.filename.split('_')[-1].split('.')[0]
            ground_truth_image = f"{dataset_name}_{test_id}_Segmentation.png"
            
            # Check if corresponding ground truth image exists
            matching_gt_image = next(
                (img for img in ground_truth_images if img.filename == ground_truth_image), None
            )
            if matching_gt_image:
                # Save test and ground truth images
                test_image.save(os.path.join(test_dir, test_image.filename))
                matching_gt_image.save(os.path.join(ground_truth_dir, ground_truth_image))
                
                # Map images with forward slashes in paths
                data.append({
                    "#": test_id,
                    "image_name": test_image.filename,
                    "test_images": f"Test_images/{test_image.filename}",
                    "ground_truth": f"Ground_truth/{ground_truth_image}"
                })
            else:
                unmapped_images.append(test_image.filename)
        logger.info(f"Mapped {len(data)} test images; {len(unmapped_images)} unmapped images found")

        # Save mapping to CSV
        df = pd.DataFrame(data, columns=["#", "image_name", "test_images", "ground_truth"])
        df.to_csv(mapping_file, index=False)
        logger.info(f"Mapping file saved at {mapping_file}")

        # Update JSON with dataset record
        dataset_info = {
            "Unique ID": generate_unique_id(),
            "Task Type": task_type,
            "Dataset Name": dataset_name,
            "Path": base_dir,
            "Number of Models": len(model_files),
            "Number of Test Images": len(test_images),
            "Number of Ground Truth Images": len(data),
            "Active": True
        }

        with open(json_file, "r+") as f:
            records = json.load(f)
            # Mark any old records with same task type and name as inactive
            for record in records:
                if record["Task Type"] == task_type and record["Dataset Name"] == dataset_name:
                    record["Active"] = False
            records.append(dataset_info)
            f.seek(0)
            json.dump(records, f, indent=4)
        logger.info(f"Dataset record updated in {json_file}")

        # Return success message with detailed info about unmapped images if any
        if unmapped_images:
            return False, f"Dataset processed successfully, but the following images were unmapped: {', '.join(unmapped_images)}"
        return True, "Dataset processed successfully."

    except Exception as e:
        logger.error("An error occurred while processing the dataset form", exc_info=True)
        return False, f"An error occurred: {str(e)}"
