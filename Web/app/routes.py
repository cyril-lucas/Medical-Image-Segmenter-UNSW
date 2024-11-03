import os
import json
import pandas as pd
import requests
from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from app.utils import (
    setup_logging,
    load_environment,
    check_essential_paths,
    get_task_types_from_data,
    get_datasets_by_task,
    get_models_by_dataset,
    verify_images_in_mapping,
    get_model_list,
    get_dataset_list
)
from datetime import datetime



main = Blueprint('main', __name__)
logger = setup_logging()
# Load environment and check paths
load_environment()
check_essential_paths(logger)

@main.route('/')
def index():
    records = []
    no_data = False
    data_file = os.path.join(os.getenv('APP_DATA_PATH'), 'data_record.json')
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
            records = [record for record in data if record.get("Active")]
        if not records:
            no_data = True
    except FileNotFoundError:
        no_data = True  # data_record.json not found
    
    return render_template('index.html', records=records, no_data=no_data)

@main.route('/get_task_types')
def get_task_types():
    task_types = get_task_types_from_data()
    return jsonify(taskTypes=list(task_types))

@main.route('/get_datasets')
def get_datasets():
    task_type = request.args.get('taskType')
    datasets = get_datasets_by_task(task_type)
    return jsonify(datasets=list(datasets))

@main.route('/get_models')
def get_models():
    dataset = request.args.get('dataset')
    task_type = request.args.get('taskType')
    models = get_models_by_dataset(task_type, dataset)
    return jsonify(models=list(models))

@main.route('/verify_images_in_mapping', methods=['POST'])
def verify_images_in_mapping():
    data = request.get_json()
    image_names = data['imageNames']
    task_type = data['taskType']
    dataset = data['dataset']
    mapping_file_path = os.path.join(os.getenv("APP_DATA_PATH"), task_type, dataset, "mapping.csv")

    try:
        df = pd.read_csv(mapping_file_path)
        missing_images = [img for img in image_names if img not in df['image_name'].values]
        return jsonify({"missingImages": missing_images})
    except FileNotFoundError:
        return jsonify({"error": "Mapping file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main.route("/folder_upload", methods=["GET", "POST"])
def folder_upload():
    logger.info("Received request at /folder_upload")
    task_types = get_task_types_from_data()  # Task types for dropdown
    ai_host = os.getenv("AI_HOST", "localhost")
    ai_port = os.getenv("AI_PORT", "3000")
    models, datasets = [], []

    if request.method == "POST":
        selected_task = request.form.get("taskType")
        selected_dataset = request.form.get("dataset")
        selected_model = request.form.get("model")
        test_folder = request.files.getlist("testFolder")

        # Filter and validate image files
        image_files = [f for f in test_folder if f.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

        # Validation checks
        if not selected_task or not selected_dataset or not selected_model:
            flash("Please select task type, dataset, and model.", "error")
            return render_template("folder_upload.html", task_types=task_types, datasets=datasets, models=models)

        if len(image_files) == 0:
            flash("Please upload a folder containing at least one image file.", "error")
            return render_template("folder_upload.html", task_types=task_types, datasets=datasets, models=models)

        # Check if images are listed in the mapping file
        mapping_file_path = os.path.join(os.getenv("APP_DATA_PATH"), selected_task, selected_dataset, "mapping.csv")
        missing_images = verify_images_in_mapping(image_files, mapping_file_path)

        if missing_images:
            flash(f"Some images are not listed in the mapping file: {', '.join(missing_images)}", "error")
            return render_template("folder_upload.html", task_types=task_types, datasets=datasets, models=models)

        # Make request to AI API /predict endpoint
        api_url = f"http://{os.getenv('AI_HOST')}:{os.getenv('AI_PORT')}/predict"
        files = [('testFolder', (file.filename, file.stream, file.mimetype)) for file in image_files]
        data = {'taskType': selected_task, 'dataset': selected_dataset, 'model': selected_model}

        try:
            response = requests.post(api_url, files=files, data=data)
            
            # Log the raw response and status code
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Raw response content: {response.content.decode('utf-8')}")
            
            # Attempt to parse JSON response
            try:
                response_data = response.json()
                logger.info(f"Parsed JSON response_data: {response_data}")
                
                if response.status_code == 200 and response_data.get('success'):
                    unique_id = response_data['unique_id']
                    return redirect(url_for('main.folder_result', unique_id=unique_id))
                else:
                    flash("An error occurred while processing your request.", "error")
            except ValueError as json_error:
                # Log JSON parsing error and raw content
                logger.error("Failed to parse JSON response")
                logger.error(f"JSON error: {json_error}")
                logger.error(f"Raw response content: {response.text}")
                flash("An error occurred while processing the response from the AI service.", "error")

        except requests.RequestException as e:
            flash("Unable to connect to the AI service.", "error")
            logger.error(f"Error contacting AI API: {e}")

    return render_template("folder_upload.html", task_types=task_types, datasets=datasets, models=models, AI_HOST=ai_host, AI_PORT=ai_port)

@main.route("/folder_result/<unique_id>")
def folder_result(unique_id):
    """Render the result page displaying records for a specific unique_id from result_record.json."""
    result_record_path = os.path.join(os.getenv("APP_RESULT_PATH"), "result_record.json")
    record = {}

    # Load data from result_record.json for the given unique_id
    if os.path.exists(result_record_path):
        with open(result_record_path, "r") as f:
            data = json.load(f)
            record = next((item for item in data if item["Unique ID"] == unique_id), {})

    if not record:
        flash("No result found for the specified ID.", "error")
        return redirect(url_for('main.index'))

    return render_template("folder_result.html", record=record)


@main.route("/img_upload", methods=["GET", "POST"])
def img_upload():
    logger.info("Received request at /img_upload")
    task_types = get_task_types_from_data()  # Task types for dropdown
    ai_host = os.getenv("AI_HOST", "localhost")
    ai_port = os.getenv("AI_PORT", "3000")
    models, datasets = [], []

    if request.method == "POST":
        # Retrieve form data and the uploaded image
        selected_task = request.form.get("taskType")
        selected_dataset = request.form.get("dataset")
        selected_model = request.form.get("model")
        test_image = request.files.get("testImage")

        # Validation checks
        if not selected_task or not selected_dataset or not selected_model:
            flash("Please select task type, dataset, and model.", "error")
            return render_template("img_upload.html", task_types=task_types, datasets=datasets, models=models)

        # Check if a valid image file is uploaded
        if not test_image or not test_image.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            flash("Please upload a valid image file in .png, .jpg, .jpeg, or .bmp format.", "error")
            return render_template("img_upload.html", task_types=task_types, datasets=datasets, models=models)

        # Check if image is listed in the mapping file
        mapping_file_path = os.path.join(os.getenv("APP_DATA_PATH"), selected_task, selected_dataset, "mapping.csv")
        missing_images = verify_images_in_mapping([test_image.filename], mapping_file_path)

        if missing_images:
            flash(f"Image is not listed in the mapping file: {', '.join(missing_images)}", "error")
            return render_template("img_upload.html", task_types=task_types, datasets=datasets, models=models)

        # Make request to AI API /predict endpoint
        api_url = f"http://{os.getenv('AI_HOST')}:{os.getenv('AI_PORT')}/predict"
        files = {'testImage': (test_image.filename, test_image.stream, test_image.mimetype)}
        data = {'taskType': selected_task, 'dataset': selected_dataset, 'model': selected_model}

        try:
            response = requests.post(api_url, files=files, data=data)
            
            # Log the raw response and status code
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Raw response content: {response.content.decode('utf-8')}")
            
            # Attempt to parse JSON response
            try:
                response_data = response.json()
                logger.info(f"Parsed JSON response_data: {response_data}")
                
                if response.status_code == 200 and response_data.get('success'):
                    unique_id = response_data['unique_id']
                    return redirect(url_for('main.img_result', unique_id=unique_id))
                else:
                    flash("An error occurred while processing your request.", "error")
            except ValueError as json_error:
                # Log JSON parsing error and raw content
                logger.error("Failed to parse JSON response")
                logger.error(f"JSON error: {json_error}")
                logger.error(f"Raw response content: {response.text}")
                flash("An error occurred while processing the response from the AI service.", "error")

        except requests.RequestException as e:
            flash("Unable to connect to the AI service.", "error")
            logger.error(f"Error contacting AI API: {e}")

    return render_template("img_upload.html", task_types=task_types, datasets=datasets, models=models, AI_HOST=ai_host, AI_PORT=ai_port)


@main.route("/img_result/<unique_id>")
def img_result(unique_id):
    """Render the result page displaying records for a specific unique_id from result_record.json."""
    result_record_path = os.path.join(os.getenv("APP_RESULT_PATH"), "result_record.json")
    record = {}

    # Load data from result_record.json for the given unique_id
    if os.path.exists(result_record_path):
        with open(result_record_path, "r") as f:
            data = json.load(f)
            record = next((item for item in data if item["Unique ID"] == unique_id), {})

    if not record:
        flash("No result found for the specified ID.", "error")
        return redirect(url_for('main.index'))

    return render_template("img_result.html", record=record)
