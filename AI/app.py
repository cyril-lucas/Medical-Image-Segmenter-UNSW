import os
import json
import logging
import shutil
from flask import Flask, request, jsonify
import random
from flask_cors import CORS

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
    test_folder_dir = os.path.join(result_dir, "test_folder")
    ground_truth_dir = os.path.join(result_dir, "ground_truth")
    model_dir = os.path.join(result_dir, "model")
    logger.info(f"unique_id: {unique_id}")

    # Create necessary directories
    os.makedirs(test_folder_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save uploaded file(s)
    for file in test_files:
        file_path = os.path.join(test_folder_dir, os.path.basename(file.filename))  # Get only the filename
        file.save(file_path)

    # Load and process mapping file if it exists
    mapping_path = os.path.join(os.getenv("APP_DATA_PATH", "/shared/data"), task_type, dataset_name, "mapping.csv")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as mapping_file:
            # Skip header row
            next(mapping_file)

            # Set of base filenames of uploaded files
            uploaded_filenames = {os.path.basename(file.filename) for file in test_files}
            
            # Process each line in the CSV
            for line in mapping_file:
                _, image_name, _, ground_truth_path = line.strip().split(",")
                ground_truth_image_name = os.path.basename(ground_truth_path)

                # Check if the uploaded file has a matching ground truth image
                if image_name in uploaded_filenames:
                    # Construct the absolute path for the ground truth source and destination
                    source_path = os.path.join(os.getenv("APP_DATA_PATH", "/shared/data"), task_type, dataset_name, ground_truth_path)
                    dest_path = os.path.join(ground_truth_dir, ground_truth_image_name)

                    # Copy the ground truth image to the destination directory
                    shutil.copy(source_path, dest_path)
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

    # Prepare JSON data for the result record
    json_data = {
        "Unique ID": unique_id,
        "Type": upload_type,
        "Test_image": test_folder_dir,
        "ground_truth": ground_truth_dir,
        "Dataset Name": dataset_name,
        "Model": model_source_path
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