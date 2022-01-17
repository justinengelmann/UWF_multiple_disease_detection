## Detection of multiple retinal diseases in ultra-widefield fundus images

The code was developed and run on Ubuntu 20.04.
You can use the ```requirements.txt``` to install the necessary packages. However, it was generated with ```pip freeze``` and might contain superfluous dependencies. We provide it to document the exact versions of all packages that were used. Alternatively, install pytorch according to the instructions at https://pytorch.org/get-started/locally/ and then run ```pip install timm sklearn matplotlib tqdm pandas numpy notebook``` which should install all necessary dependencies. This should take a few minutes.

In the code, change ```PATH/TO/RAW/DATA/``` to the path containing the unzipped raw data, as it was originally provided by Dr. Hiroki Masumoto, and ```PATH/TO/PREPROCESSED/DATA/``` to the path where you wish to store the processed data. Unfortunately, we are unable to redistribute the data ourselves.

Then run ```TOP_preprocessing.py``` to preprocess the data and use ```TOP_Datasplit.ipynb``` to generate the train, validation and test sets. You should now be able to train a model with ```TOP_Training.ipynb```. 

A recent NVIDIA GPU is recommended for model training. 
