FINAL_CODE
├── AI
│ ├── data
│ │ ├── ground_truth
│ │ ├── sampled
│ │ ├── upload
│ │ └── metrics_results.csv
│ ├── guided_diffusion
│ ├── model
│ │ ├── emasavedmodel_0.9999_000000.pt
│ │ ├── emasavedmodel_0.9999_000002.pt
│ │ ├── savedmodel000000.pt
│ │ └── savedmodel000002.pt
│ ├── Dockerfile
│ ├── requirement.txt
│ ├── runcommand.txt
│ ├── segmentation_env_PerClass.py
│ ├── segmentation_env.py
│ └── segmentation_sample.py
├── Web
│ ├── **pycache**
│ ├── AI
│ ├── app
│ │ ├── **pycache**
│ │ ├── routes.py
│ │ └── utils.py
│ ├── static
│ │ ├── css
│ │ │ └── main.css
│ │ └── images
│ │ ├── Background.jpg
│ │ ├── Background.png
│ │ └── favicon.ico
│ ├── templates
│ │ ├── folder_result.html
│ │ ├── folder_upload.html
│ │ ├── img_result.html
│ │ ├── img_upload.html
│ │ └── index.html
│ ├── .dockerignore
│ ├── app.log
│ ├── Dockerfile
│ ├── main.py
│ ├── requirements.txt
├── .env
├── docker-compose.yml
└── Readme.md

TO DO
--- store ground truth in AI/data/ground_truth/dataset_Name(ISIC)/

RUN Method
docker-compose up --build

docker-compose down

FINAL_CODE
├── AI
│ ├── app.py # Main script for the AI service
│ ├── Dockerfile # Dockerfile for building the AI service
│ └── requirements.txt # Dependencies for AI service
├── Web
│ ├── app # Flask app directory
│ ├── Dockerfile # Dockerfile for building the Web service
│ └── requirements.txt # Dependencies for Web service
├── shared
│ ├── data # Centralized data directory for uploads, ground truth, results
│ │ ├── ground_truth # Ground truth images for datasets
│ │ ├── result # Result folders created per unique upload
│ │ ├── upload_log.csv # CSV log of uploads
│ │ └── metrics_results.csv # Additional metrics (if needed)
│ ├── model # Model directory shared between services
│ └── logs # Centralized log directory
│ └── app.log # Main log file for both services
├── docker-compose.yml # Docker Compose file for both services
└── .env # Environment configuration file

Setup dataset
cd Dataset_setup
pip install -r requirements.txt
python data_setup.py

Docker setup
docker system prune -a --volumes
docker-compose down  
 docker-compose up --build

first docker setup will take couple of minutes
