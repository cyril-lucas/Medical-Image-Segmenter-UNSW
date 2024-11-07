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
Reset docker : docker system prune -a --volumes
Start Docker : docker-compose up --build
Stop Docker : docker-compose down
Setup Dataset: docker-compose run data_setup

"GET /img_result/6043455593
