# Dynamic-budget Superpixel Active Learning
This repository is implemented with Keras.

## Requirements
- Python 3.11+
- NVIDIA GPU (>= 20GiB VRAM)
- CUDA 12
- Install **src/requirements.txt**

## Datasets
- **Cityscapes** 
  - **leftImg8bit_trainvaltest** (RGB images folder structure)
    - leftImg8bit 
      - train
      - val
      - test
  - **gtFine_trainvaltest** (Labels folder structure)
    - gtFine
      - train
      - val
      - test
  - Place provided **cityscapes_filelist/train.txt** in 
    - leftImg8bit_trainvaltest/leftImg8bit/train 
  - Place provided **cityscapes_filelist/val.txt** in 
    - leftImg8bit_trainvaltest/leftImg8bit/val 
  
  
- **Nassar 2020** 
  - train (Dataset folder structure)
    - data (RGB images)
    - data_list 
      - train.txt
      - val.txt
      - test.txt
    - mask (Labels)

The Nassar 2020 dataset can be accessed with the link below:
https://www.kaggle.com/datasets/yueminwang/nassar-2020

## Config Files
Copy and rename  **cfg/config_template.yml** to **cfg/config.yml**

This file controls experiment parameters.
- **cfg/config_template.yml**
  - **runner**: set name to one below to specify which runner to use
    - tf_bnn_cityscape: train a model with full cityscapes
    - tf_bnn_cityscape_random: random whole-image query AL on cityscapes
    - tf_bnn_cityscape_sp_active_learn: superpixel AL on cityscapes
    - tf_bnn_weed_map: train a model with full nassar2020
    - tf_bnn_cityscape_random: random whole-image query AL on nassar2020 
    - tf_bnn_weed_map_sp_active_learn: superpixel AL on nassar2020 
  - **mode**: set to train to run experiments; 
    - only set to test after tf_bnn_cityscape or tf_bnn_weed_map to test full baselines. 
  - **test_vis**: true for AL visualizations 
  - **datasets**
    - **nassar2020**
      - **dir**: fill with the root dir of nassa2020 dataset
    - **cityscapes**
      - **dir**: fill with the root dir of cityscapes dataset
    - **active_learn**
      - **init_size**: initialization step size, fraction of whole dataset 
      - **step_size**: AL step size, fraction of whole dataset 
      - **pixel_mode**: 0 for static-budget, 1 for dynamic-budget
      - **pixel_percent**: static-budget fraction per image; 
        - this controls the percentage of superpixels in an image will be labelled; only for static-budget 
      - **sp_param**: number of superpixels per image
      - **experiment**
        - **logger**: neptune; using neptune to record experiments
        - **projectName**: neptune project name
        - **experimentName**: neptune experiment name
        - **tags**: [list of tags for experiments]
        
copy and rename **cfg/private_template.yml** to **cfg/private.yml**
- **cfg/private_template.yml**
  - **neptune**
    - **api_key**: paste neptune api key here 
    - **username**: neptune user name
    - **project_name**: neptune project name
  - **rootDataDir**: local directory where checkpoints, models, and visualizations will be saved


## Run Experiments
1. Setup environments
2. Complete and rename configuration files: **cfg/config.yml** and **cfg/private_template.yml**
3. Run experiments
    - Fully labelled baselines:
      - set **runner** to tf_bnn_cityscape or tf_bnn_weed_map 
      - set **mode** to train in **cfg/config.yml**
      - run: python main.py
      - after training is finished
      - set **mode** to test in **cfg/config.yml** 
      - run: python main.py
    - Superpixel AL:
      - _(a)._ config files
        - set **init_size** and **step_size**
          - set to 0.1 to label 10% of images per step
          - set to 0.0169 to label 50 image per step for cityscapes
          - set to 0.0259 to label 50 images per step for nassar 2020
        - set **pixel_mode** and **pixel_percent**(static-budget only) 
        - set **sp_param** 
      - _(b)._ generate superpixel maps
        - run: python sp_gen_nassar.py
          - this generates sp_map for nassar 2020 with **sp_param**
        - run: python sp_gen_cityscapes.py
          - this generates sp_map for cityscapes with **sp_param** 
      - _(c)._ run experiment
        - cityscapes experiment
          - set **runner** to tf_bnn_cityscape_sp_active_learn
          - run: python main.py
        - nassar 2020 experiment
          - set **runner** to tf_bnn_weed_map_sp_active_learn
          - run: python main.py
