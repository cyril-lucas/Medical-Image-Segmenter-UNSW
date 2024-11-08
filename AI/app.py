import os
import json
import logging
import shutil
import subprocess
from flask import Flask, request, jsonify
import random
from flask_cors import CORS
import csv


app = Flask(__name__)
CORS(app) 

# Configure logger
log_dir = os.getenv("APP_LOG_PATH", "/shared/logs")
os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists
log_file_path = os.path.join(log_dir, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Helper function to generate a unique 10-digit ID
def generate_unique_id():
    return str(random.randint(1000000000, 9999999999))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    logger.info("Received request at /predict")

    # Determine if it's a folder or single image upload
    if 'testFolder' in request.files:
        upload_type = "Folder"
        test_files = request.files.getlist("testFolder")
    elif 'testImage' in request.files:
        upload_type = "Image"
        test_files = [request.files.get("testImage")]
    else:
        logger.error("Missing files or form data in the request.")
        return jsonify({"error": "Missing files or form data in the request."}), 400

    # Validate form data
    if not request.form.get("taskType") or not request.form.get("dataset") or not request.form.get("model"):
        logger.error("Missing form data in the request.")
        return jsonify({"error": "Missing form data in the request."}), 400

    # Extract form data
    task_type = request.form.get("taskType")
    dataset_name = request.form.get("dataset")
    model_name = request.form.get("model")

    # Generate unique ID and create result directory
    unique_id = generate_unique_id()
    result_dir = os.path.join(os.getenv("APP_RESULT_PATH", "/shared/result"), unique_id)
    test_folder_dir = os.path.join(result_dir, "Test_images")
    ground_truth_dir = os.path.join(result_dir, "Ground_truth")
    model_dir = os.path.join(result_dir, "model")
    sampled_dir = os.path.join(result_dir, "sampled")
    logger.info(f"unique_id: {unique_id}")

    # Create necessary directories
    os.makedirs(test_folder_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(sampled_dir, exist_ok=True)


    # Save uploaded file(s)
    for file in test_files:
        file_path = os.path.join(test_folder_dir, os.path.basename(file.filename))  # Get only the filename
        file.save(file_path)



    # Load and process mapping file if it exists
    mapping_path = os.path.join(os.getenv("APP_DATA_PATH", "/shared/data"), task_type, dataset_name, "mapping.csv")
    filtered_mapping_path = os.path.join(result_dir, "mapping.csv")
    # Check if the mapping file exists
    if os.path.exists(mapping_path):
        # Open the original mapping file and the filtered output file
        with open(mapping_path, "r") as mapping_file, open(filtered_mapping_path, "w", newline="") as filtered_file:
            reader = csv.reader(mapping_file)
            writer = csv.writer(filtered_file)
            
            # Define the expected header and write it to the filtered file
            header = ["#", "image_name", "test_images", "ground_truth"]
            writer.writerow(header)

            # Create a set of uploaded image filenames without paths (e.g., {"ISIC_0012169.jpg", "ISIC_0012236.jpg"})
            uploaded_filenames = {os.path.basename(file.filename) for file in test_files}
            
            # Skip the header of the original mapping file
            next(reader)
            
            # Process each row in the original mapping file
            for row in reader:
                _, image_name, test_image_path, ground_truth_path = row

                # Check if the image_name is in the set of uploaded filenames
                if image_name in uploaded_filenames:
                    # Copy the ground truth image to the destination directory
                    ground_truth_image_name = os.path.basename(ground_truth_path)
                    source_path = os.path.join(os.getenv("APP_DATA_PATH", "/shared/data"), task_type, dataset_name, ground_truth_path)
                    dest_path = os.path.join(ground_truth_dir, ground_truth_image_name)
                    shutil.copy(source_path, dest_path)

                    # Write the matching row to the filtered mapping file
                    writer.writerow(row)
    else:
        logger.error(f"Mapping file not found at {mapping_path}")
        return jsonify({"error": "Mapping file not found"}), 404

    # Copy the selected model to model directory
    model_source_path = os.path.join(os.getenv("APP_DATA_PATH", "/shared/data"), task_type, dataset_name, "model", model_name)
    if os.path.exists(model_source_path):
        shutil.copy(model_source_path, os.path.join(model_dir, model_name))
    else:
        logger.error(f"Model file not found at {model_source_path}")
        return jsonify({"error": "Model file not found"}), 404
    

   # Run segmentation_sample.py with dynamically passed arguments
    sample_process = subprocess.run([
        "python", "scripts/segmentation_sample.py",
        "--data_name", dataset_name,
        "--data_dir", result_dir,
        "--out_dir", sampled_dir,
        "--model_path", os.path.join(model_dir, model_name)
    ])

    # Check if segmentation_sample.py ran successfully
    if sample_process.returncode == 0:
        # Run segmentation_eval.py and capture the output if sampling was successful
        eval_output = subprocess.check_output([
            "python", "scripts/segmentation_eval.py",
            "--inp_pth", sampled_dir,
            "--out_pth", ground_truth_dir
        ], text=True)
    else:
        # Log an error if segmentation_sample.py failed
        logger.error("segmentation_sample.py failed to execute successfully. Skipping segmentation_eval.py.")
        return jsonify({"error": "segmentation_sample.py failed to execute successfully."}), 500

    # Parse the metrics from the output of segmentation_eval.py
    metrics = {}
    for line in eval_output.strip().splitlines():
        if line.startswith("IoU:"):
            metrics["IoU"] = round(float(line.split(":")[1].strip()), 5)
        elif line.startswith("Dice Coefficient:"):
            metrics["Dice Score"] = round(float(line.split(":")[1].strip()), 5)
        elif line.startswith("Accuracy:"):
            metrics["Accuracy"] = round(float(line.split(":")[1].strip()), 5)
        elif line.startswith("Sensitivity:"):
            metrics["Sensitivity"] = round(float(line.split(":")[1].strip()), 5)
        elif line.startswith("Specificity:"):
            metrics["Specificity"] = round(float(line.split(":")[1].strip()), 5)
        elif line.startswith("F1 Score:"):
            metrics["F1 Score"] = round(float(line.split(":")[1].strip()), 5)

    # Prepare JSON data for the result record
    json_data = {
        "Unique ID": unique_id,
        "Type": upload_type,
        "Test_image": test_folder_dir,
        "ground_truth": ground_truth_dir,
        "Dataset Name": dataset_name,
        "Model": model_source_path,
        "sampled_path": sampled_dir,
        **metrics  # Add the parsed metrics to the JSON data

    }

    # Append to result_record.json or create it if it doesn't exist
    result_record_path = os.path.join(os.getenv("APP_RESULT_PATH", "/shared/result"), "result_record.json")
    if not os.path.exists(result_record_path):
        with open(result_record_path, "w") as f:
            json.dump([json_data], f, indent=4)
    else:
        with open(result_record_path, "r+") as f:
            data = json.load(f)
            data.append(json_data)
            f.seek(0)
            json.dump(data, f, indent=4)

    logger.info(f"Data successfully saved for Unique ID: {unique_id}")
    return jsonify({"success": True, "unique_id": unique_id}), 200

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check endpoint accessed")
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host=os.getenv('AI_HOST', '0.0.0.0'), port=int(os.getenv('AI_PORT', 3000)), debug=(os.getenv("ENV") == "development"))
