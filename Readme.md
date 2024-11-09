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

(finalproj_train) PS D:\Personal\UNSW\Ongoing\COMP9900\Code\capstone-project-2024-t3-9900t15apotatoes> git status  
On branch csl_working  
Changes not staged for commit:  
 (use "git add/rm <file>..." to update what will be committed)  
 (use "git restore <file>..." to discard changes in working directory)  
 modified: AI/Dockerfile
modified: AI/app.py
modified: AI/guided_diffusion/**pycache**/**init**.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/dist_util.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/dpm_solver.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/fp16_util.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/gaussian_diffusion.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/isicloader.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/logger.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/losses.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/nn.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/resample.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/respace.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/script_util.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/train_util.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/unet.cpython-39.pyc
modified: AI/guided_diffusion/**pycache**/utils.cpython-39.pyc
deleted: AI/guided_diffusion/bratsloader.py
deleted: AI/guided_diffusion/custom_dataset_loader.py
modified: AI/guided_diffusion/dist_util.py
modified: AI/guided_diffusion/isicloader.py
modified: AI/guided_diffusion/train_util.py
modified: AI/guided_diffusion/unet.py
modified: AI/requirement.txt
deleted: AI/scripts/segmentation_env.py
deleted: AI/scripts/segmentation_env_PerClass.py
modified: AI/scripts/segmentation_sample.py

Untracked files:
(use "git add <file>..." to include in what will be committed)
AI/scripts/arguments.json
AI/scripts/segmentation_eval.py
Model_train/

no changes added to commit (use "git add" and/or "git commit -a")
(finalproj_train) PS D:\Personal\UNSW\Ongoing\COMP9900\Code\capstone-project-2024-t3-9900t15apotatoes> git add .  
(finalproj_train) PS D:\Personal\UNSW\Ongoing\COMP9900\Code\capstone-project-2024-t3-9900t15apotatoes> git commit -m "model setup"  
[csl_working d62ed4c] model setup  
 30 files changed, 616 insertions(+), 840 deletions(-)  
 delete mode 100644 AI/guided_diffusion/bratsloader.py  
 delete mode 100644 AI/guided_diffusion/custom_dataset_loader.py
create mode 100644 AI/scripts/arguments.json
delete mode 100644 AI/scripts/segmentation_env.py
delete mode 100644 AI/scripts/segmentation_env_PerClass.py
create mode 100644 AI/scripts/segmentation_eval.py
create mode 100644 Model_train/ISIC/segmentation_train.py
(finalproj_train) PS D:\Personal\UNSW\Ongoing\COMP9900\Code\capstone-project-2024-t3-9900t15apotatoes> git push origin csl_working  
Enumerating objects: 48, done.  
Counting objects: 100% (48/48), done.  
Delta compression using up to 20 threads  
Compressing objects: 100% (32/32), done.
Writing objects: 100% (34/34), 95.30 KiB | 5.61 MiB/s, done.
Total 34 (delta 9), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (9/9), completed with 9 local objects.
To https://github.com/unsw-cse-comp99-3900/capstone-project-2024-t3-9900t15apotatoes.git
4fecc11..d62ed4c csl_working -> csl_working
(finalproj_train) PS D:\Personal\UNSW\Ongoing\COMP9900\Code\capstone-project-2024-t3-9900t15apotatoes>
