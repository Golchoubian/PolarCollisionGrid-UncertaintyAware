
# Uncertinty Aware Polar Collision Grid (UAW-PCG)

This repository contains the code for the uncertainty-aware version of the polar collision grid trajectory prediction model presented in the paper ["paper name"](arXiv address). 

## Overview

The presented uncertainty-aware pedestrian trajectory prediction model is based on the Polar Collision Grid (PCG) model introduced in our ITSC2023 paper. However, we have enhanced this model by training it with a novel uncertainty-aware loss function. This modification aims to improve the accuracy of predicting the covariance of future positions within the forecasted distribution. The original PCG model, trained solely with the Negative Log Likelihood loss, tends to generate overconfident predictions. To address this issue, we introduced an uncertainty-loss component which penalizes the mahalanobis distance of the ground truth position from the predicted distribution (Point2Dist loss). This addition has proven to enhance the performance of our model, as detailed in our paper.

<div style="display: inline-block;">
    <img src="https://github.com/Golchoubian/PolarCollisionGrid-UncertaintyAware/blob/master/figure/PCG.png?raw=true" alt="PCG" width="400" hspace="50"> 
   <img src="https://github.com/Golchoubian/PolarCollisionGrid-UncertaintyAware/blob/master/figure/UAW-PCG.png?raw=true" alt="UAW-PCG" width="400">
</div>



## Setup

Create a conda environmnet using python version 3.9, and install the required python packages
```bash
conda create --name PCG python=3.9
conda activate PCG
pip install -r requirements.txt
```
Install pytorch version 2.2.1 using the instructions [here](https://pytorch.org/get-started/locally/)



## Dataset

The [HBS dataset](https://leopard.tu-braunschweig.de/receive/dbbs_mods_00069907), which includes trajectories of both pedestrians and vehicles collected from a shared space, is used for training and testing our data-driven trajectory prediction model. The initial dataset, stored as `hbs.csv` in the `Data` folder, undergoes preprocessing. This involves adapting it to a format compatible with our code, utilizing the functions available in the `datalader.py` file. Subsequently, the data is partitioned into train and test sets, which are already stored in the `Data` folder.

## Model training

The Polar Collision Grid model can undergo training in either its original format, utilizing only NLL loss, or in the uncertainty-aware version with the novel combination loss, adjustable through the `uncertainty-aware` argument in the `train.py` file. In this improved code version, we have also incorporated the option to train the model without teacher forcing. By setting the `teacher-forcing` argument to `False`, the model relies on its own predicted outputs even during the training phase, rather than the ground truth for the prediction length. The remaining functionalities remain consistent with those described in the original PCG model's repository.

Additionally, we have developed another uncertainty-aware loss function, termed Dist2Dist loss, which aims to minimize the distance between the prediction distribution and a ground truth distribution derived from a Kalman filter applied to the ground truth position. However, the results reported in the paper are based on the Point2Dist loss function.

In both scenarios, the code provides the option to include the covariance matrix as an additional input to the model. This can be achieved by setting the `input_size` argument to 6 in the `train.py` file.

The same training process can be applied to other baseline methods (Social LSTM, Vanilla LSTM) available in the code by adjusting the `model` argument.


## Model evaluation

Our trained models are stored in folders named relative to their corresponding models under the `Store_Results\model` directory. By executing the `test.py` script, the model saved in these directories will be loaded and tested on the test set. Depending on the chosen method, the `epoch` argument associated with the saved model in its folder should be adjusted. The terminal will display the performance of the saved model for the defined evaluation metrics, and the outputted trajectories for the test set will be saved as a `test_result.pkl` file in the `Store_Results/plot/test` directory.

For the results reported in the paper, the test files are stored in the method's associated folder within the `Store_Results/plot/test` directory. These files can be utilized to run the `visualization.py` script and generate the table results in the paper by executing the `TableResults.py` file for the selected method.

## Visualization
The predicted trajectories can be visualized for selected samples in the test set by running the `visualization.py` file. The resulting figures are stored in the `Store_Results\plot\test\plt` directory.

## Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@inproceedings{golchoubian2023polar,
  title={Polar Collision Grids: Effective Interaction Modelling for Pedestrian Trajectory Prediction in Shared Space Using Collision Checks},
  author={Golchoubian, Mahsa and Ghafurian, Moojan and Dautenhahn, Kerstin and Azad, Nasser Lashgarian},
  booktitle={2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)},
  pages={791--798},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgment
This project is builds upon the codebase from social-lstm repsitory,
developed by [quancore](https://github.com/quancore/social-lstm) as a pytorch implementation of the Social LSTM model proposed by [Alahi et al](https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf).
The Social LSTM model itself is also used as a baseline for comparison with our propsed CollisionGrid model.



