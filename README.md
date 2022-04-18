# 3D-Object-Tracking-And-Motion-Forecasting

1. For running the Improved Object Tracking experiments with different cost functions, change the default cost_type in the cost_matrix() function definition on line 58 of the tracker.py file.

   cost_type=0 for only IoU cost <br>
   cost_type=1 for only geometric distance cost <br>
   cost_type=2 for only motion feature cost <br>
   cost_type=3 for combined IoU and motion feature cost <br>


   Then run the following: 
   ```
   python -m tracking.main track --dataset_path=<path> --tracker_associate_method=hungarian
   python -m tracking.main evaluate
   python -m tracking.main visualize

   ```

2. For running the Improved Object Tracking experiments with occlusion handling, change the default iou_th in the is_connected() method definititon on line 51 of the types.py file.

   iou_th=0.1 gave better MOTA and MOTP than iou_th=0.5 for the 12 validation sequences we used.

   <br>

# CSC490H1: Making Your Self-driving Car Perceive the World

This repository contains the starter code for CSC490H1:
Making Your Self-driving Car Perceive the World.

## Getting started

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

   ```bash
   curl 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh' > Miniconda.sh
   bash Miniconda.sh
   rm Miniconda.sh
   ```

2. Close and re-open your terminal session.

3. Change directories (`cd`) to where you cloned this repository.

4. Create a new conda environment:

   ```bash
   conda env create --file environment.yml
   ```

   To update the conda environment with new packages:

   ```bash
   conda env update --file environment.yml
   ```

5. Activate your new environment:

   ```bash
   conda activate csc490
   ```

6. Download [PandaSet](https://scale.com/resources/download/pandaset).
   After submitting your request to download the dataset, you will receive an
   email from Scale AI with instructions to download PandaSet in three parts.
   Download Part 1 only. After you have downloaded `pandaset_0.zip`,
   unzip the dataset as follows:

   ```bash
   unzip pandaset_0.zip -d <your_path_to_dataset>
   ```
