import os
import json
import pandas as pd
import requests
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, flash, send_file, send_from_directory
from app.utils import (
    setup_logging,
    load_environment,
    check_essential_paths,
    get_task_types_from_data,
    get_datasets_by_task,
    get_models_by_dataset,
    verify_images_in_mapping,
    extract_id,
)
from app.data_setup import process_dataset_form  # Assuming process_data_setup_form exists in addon/data_setup.py
from datetime import datetime
from fpdf import FPDF


main = Blueprint('main', __name__)
logger = setup_logging()
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
    


@main.route('/datasetup', methods=['GET', 'POST'])
def datasetup():
    message = ""
    if request.method == 'POST':
        # Extract form data
        task_type = request.form.get('taskType')
        dataset_name = request.form.get('datasetName')
        model_files = request.files.getlist('modelFiles')
        test_images = request.files.getlist('testImages')
        ground_truth_images = request.files.getlist('groundTruthImages')
        
        # Pass data to process function and get response
        success, message = process_dataset_form(task_type, dataset_name, model_files, test_images, ground_truth_images)
        
    # Render the template with an inline message
    return render_template('data_setup.html', message=message)

# @main.route('/data_setup', methods=['GET', 'POST'])
# def data_setup():
#     json_file = os.path.join(os.getenv("APP_DATA_PATH"), "data_record.json")

#     if request.method == 'POST':
#         task_type = request.form.get('taskType')
#         dataset_name = request.form.get('datasetName')
#         model_files = request.files.getlist('modelFiles')
#         test_images = request.files.getlist('testImages')
#         ground_truth_images = request.files.getlist('groundTruthImages')

#         logger.info("Received POST request for data setup")
#         logger.debug(f"Task Type: {task_type}")
#         logger.debug(f"Dataset Name: {dataset_name}")
#         logger.debug(f"Model Files: {[f.filename for f in model_files]}")
#         logger.debug(f"Test Images: {[f.filename for f in test_images]}")
#         logger.debug(f"Ground Truth Images: {[f.filename for f in ground_truth_images]}")

#         # Process the form and capture any error messages
#         try:
#             result_message = process_data_setup_form(
#                 task_type, dataset_name, model_files, test_images, ground_truth_images, json_file
#             )
#             logger.info("Data setup processing completed")
#             flash(result_message)
#         except Exception as e:
#             logger.error("An error occurred during data setup processing", exc_info=True)
#             flash(f"Error: {str(e)}")
        
#         return redirect(url_for('main.data_setup'))

#     task_types = ["Skin_Lesion", "Multi-Organ_Segmentation", "Other"]
#     return render_template('data_setup.html', task_types=task_types)

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
                    return redirect(url_for('main.result', unique_id=unique_id))
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

# @main.route("/folder_result/<unique_id>")
# def folder_result(unique_id):
#     """Render the result page displaying records for a specific unique_id from result_record.json."""
#     result_record_path = os.path.join(os.getenv("APP_RESULT_PATH"), "result_record.json")
#     record = {}

#     # Load data from result_record.json for the given unique_id
#     if os.path.exists(result_record_path):
#         with open(result_record_path, "r") as f:
#             data = json.load(f)
#             record = next((item for item in data if item["Unique ID"] == unique_id), {})

#     if not record:
#         flash("No result found for the specified ID.", "error")
#         return redirect(url_for('main.index'))

#     return render_template("folder_result.html", record=record)


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
                    return redirect(url_for('main.result', unique_id=unique_id))
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


# @main.route("/img_result/<unique_id>")
# def img_result(unique_id):
#     """Render the result page displaying records for a specific unique_id from result_record.json."""
#     result_record_path = os.path.join(os.getenv("APP_RESULT_PATH"), "result_record.json")
#     record = {}

#     # Load data from result_record.json for the given unique_id
#     if os.path.exists(result_record_path):
#         with open(result_record_path, "r") as f:
#             data = json.load(f)
#             record = next((item for item in data if item["Unique ID"] == unique_id), {})

#     if not record:
#         flash("No result found for the specified ID.", "error")
#         return redirect(url_for('main.index'))

#     return render_template("img_result.html", record=record)


@main.route('/result/<result_id>')
def result(result_id):
    RESULT_PATH = os.getenv("APP_RESULT_PATH", "/shared/result")
    RESULT_JSON_PATH = os.path.join(RESULT_PATH, "result_record.json")

    with open(RESULT_JSON_PATH, "r") as f:
        results = json.load(f)

    # Find the result entry with the specified result_id
    result = next((r for r in results if r["Unique ID"] == result_id), None)
    if not result:
        return "Result not found", 404
    
    # Get paths for images based on the matched IDs
    test_images = {
        extract_id(file, r"ISIC_(\d+)\.jpg"): f"{result_id}/Test_images/{file}"
        for file in os.listdir(f"{RESULT_PATH}/{result_id}/Test_images")
    }
    ground_truth_images = {
        extract_id(file, r"ISIC_(\d+)_Segmentation\.png"): f"{result_id}/Ground_truth/{file}"
        for file in os.listdir(f"{RESULT_PATH}/{result_id}/Ground_truth")
    }
    sampled_images = {
        extract_id(file, r"(\d+)_output_ens\.jpg"): f"{result_id}/sampled/{file}"
        for file in os.listdir(f"{RESULT_PATH}/{result_id}/sampled")
    }

    # Align images by matching IDs
    matched_images = [
        {
            "test_image": url_for('main.serve_result_file', subpath=test_images[img_id]),
            "ground_truth_image": url_for('main.serve_result_file', subpath=ground_truth_images[img_id]),
            "sampled_image": url_for('main.serve_result_file', subpath=sampled_images[img_id])
        }
        for img_id in test_images
        if img_id in ground_truth_images and img_id in sampled_images
    ]

    # Pass matched images and scores to the template
    return render_template(
        "result.html",
        matched_images=matched_images,
        scores={
            "F1_Score": result.get("F1_Score"),
            "Specificity": result.get("Specificity"),
            "Sensitivity": result.get("Sensitivity"),
            "Accuracy": result.get("Accuracy"),
            "Dice_Score": result.get("Dice_Score"),
            "IoU": result.get("IoU")
        },
        result_id=result_id
    )



@main.route('/download_pdf/<result_id>')
def download_pdf(result_id):
    RESULT_PATH = os.getenv("APP_RESULT_PATH", "/shared/result")
    RESULT_JSON_PATH = os.path.join(RESULT_PATH, "result_record.json")
    
    # Read result_record.json within the download_pdf route
    with open(RESULT_JSON_PATH, "r") as f:
        results = json.load(f)

    result = next((r for r in results if r["Unique ID"] == result_id), None)
    if not result:
        return "Result not found", 404

    # Paths to local image files
    test_images = {
        extract_id(file, r"ISIC_(\d+)\.jpg"): os.path.join(RESULT_PATH, result_id, "Test_images", file)
        for file in os.listdir(f"{RESULT_PATH}/{result_id}/Test_images")
    }
    ground_truth_images = {
        extract_id(file, r"ISIC_(\d+)_Segmentation\.png"): os.path.join(RESULT_PATH, result_id, "ground_truth", file)
        for file in os.listdir(f"{RESULT_PATH}/{result_id}/ground_truth")
    }
    sampled_images = {
        extract_id(file, r"(\d+)_output_ens\.jpg"): os.path.join(RESULT_PATH, result_id, "sampled", file)
        for file in os.listdir(f"{RESULT_PATH}/{result_id}/sampled")
    }

    # Align images by matching IDs
    matched_images = [
        {
            "test_image": test_images[img_id],
            "ground_truth_image": ground_truth_images[img_id],
            "sampled_image": sampled_images[img_id]
        }
        for img_id in test_images
        if img_id in ground_truth_images and img_id in sampled_images
    ]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_font("Arial", "B", 14)

    # Title for Scores Section
    pdf.cell(0, 10, "Segmentation Scores", ln=True, align="C")
    pdf.ln(5)
    
    # Center the table on the page
    table_x = (210 - 80) / 2  # Center the 80mm wide table on an A4 page (210mm wide)
    pdf.set_xy(table_x, pdf.get_y())

    # Create table headers
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "Metric", border=1, align="C")
    pdf.cell(40, 10, "Score", border=1, align="C")
    pdf.ln()

    # Table content
    pdf.set_font("Arial", "", 12)
    score_keys = ["F1_Score", "Specificity", "Sensitivity", "Accuracy", "Dice_Score", "IoU"]
    for key in score_keys:
        pdf.set_x(table_x)
        pdf.cell(40, 10, key, border=1, align="C")
        pdf.cell(40, 10, str(result.get(key, 'N/A')), border=1, align="C")
        pdf.ln()  # Move to the next row

    pdf.ln(10)  # Add space below the scores section

    # Image Table Section Title
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Detailed Comparison", ln=True, align="C")
    pdf.set_font("Arial", "B", 12)
    
    # Adjusted Table Headers for Images
    headers = ["Test Image", "Ground Truth", "Sampled Image"]
    cell_width = 60  # Slightly larger width to center the table
    for header in headers:
        pdf.cell(cell_width, 10, header, border=1, align="C")
    pdf.ln()  # Move to the next row

    # Image Rows
    pdf.set_font("Arial", "", 10)
    padding = 5  # Padding around each image
    image_height = 60  # Reduced image height to fit with padding
    for images in matched_images:
        # Check if a new page is needed for the next row
        if pdf.get_y() + image_height + 2 * padding > pdf.page_break_trigger:
            pdf.add_page()
            # Reprint the header row on the new page
            for header in headers:
                pdf.cell(cell_width, 10, header, border=1, align="C")
            pdf.ln()  # Move to the next row

        y_start = pdf.get_y()  # Starting y position for images in the row

        # Reserve space for each image and add them using specific coordinates
        pdf.cell(cell_width, image_height + 2 * padding, "", border=1)  # Reserve space for Test Image
        pdf.cell(cell_width, image_height + 2 * padding, "", border=1)  # Reserve space for Ground Truth
        pdf.cell(cell_width, image_height + 2 * padding, "", border=1)  # Reserve space for Sampled Image
        pdf.ln()  # Move to the next row

        # Adjust x and y coordinates for each image to center them within cells
        x_start = 15  # Left margin offset to ensure images stay within page
        pdf.image(images["test_image"], x=x_start + padding, y=y_start + padding, w=cell_width - 2 * padding, h=image_height)
        pdf.image(images["ground_truth_image"], x=x_start + cell_width + padding, y=y_start + padding, w=cell_width - 2 * padding, h=image_height)
        pdf.image(images["sampled_image"], x=x_start + 2 * cell_width + padding, y=y_start + padding, w=cell_width - 2 * padding, h=image_height)

    # Save and return the PDF
    pdf_file_path = f"/tmp/Segmentation_Result_{result_id}.pdf"
    pdf.output(pdf_file_path)
    return send_file(pdf_file_path, as_attachment=True, download_name=f"Segmentation_Result_{result_id}.pdf")

@main.route('/shared/result/<path:subpath>')
def serve_result_file(subpath):
    result_path = os.getenv("APP_RESULT_PATH", "/shared/result")
    return send_from_directory(result_path, subpath)