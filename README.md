# Mini competition at Data Science Retreat Batch 23

This repo represents the submission for the mini competition at [Data Science Retreat](#https://www.datascienceretreat.com/)
during Batch 23. The team consists of Michael Drews, Tomasz Iżycki, and Christopher Dick (the owner of this repo). The 
task is to predict the sales of a given store and date. The datasets are based on the kaggle competition 
"[Rossmann Store Sales](#https://www.kaggle.com/c/rossmann-store-sales)". For more information on the competition, 
have a look at the original repo [here](#https://github.com/ADGEfficiency/minicomp-rossman).


## SETUP
All code is written in Python with standard data science libraries. We recommend the use of 
[Anaconda](#https://www.anaconda.com/) to set up the environment.

### Repository
 Clone this repository to your computer with:
 ```shell script
$ git clone https://github.com/ChristopherSD/dsr-minicomp.git
```
 
### Virtual Environment + Requirements
Change to the folder of your newly cloned repository. Create a new virtual environment
and install all necessary requirements. Activate the new environment.

To install the requirements:

**Conda (recommended)**

To create a new virtual environment and install all necessary requirements:
````shell script
$ conda env create -f environment.yml
````

**PiP**

To install the requirements with pip:
```shell script
$ pip install -r requirements.txt
```

## USAGE
To use the standard test data set run:
```shell script
$ python predict.py
```

To predict prices from a given dataset (*.csv file), run:
```shell script
$ python predict.py --file=path/to/test/file.csv
```