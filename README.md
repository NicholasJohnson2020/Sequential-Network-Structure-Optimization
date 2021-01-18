# Sequential-Network-Structure-Optimization

This repository contains the code used to construct the simulation environment and run the experiments presented in the paper Sequential Stochastic Network Structure Optimization with Applications To Addressing Canada's Obesity
Epidemic.

## Dependencies

- python 3.8.5
- numpy 1.19.2
- pandas 1.2.0
- matplotlib 3.3.2
- networkx 2.5
- scipy 1.5.2
- scikit-learn 0.23.2

## Usage

### Data

The 2015/2016 Canadian Community Health Survey: Public Use Microdata File comprised the raw data employed for this project. It can be requested (free of charge) at https://www150.statcan.gc.ca/n1/en/catalogue/82M0013X2019001.

The raw data was processed using the script src/data_processing_script.py. This script can be executed using the following command:

```sh
python data_processing_script.py
```

Note that to execute this script, the user must separately download the raw data (the raw data was omitted as the file is quite large). The output of running
this script is stored in src/data/processed_data.csv.

### Running Experiments

### Plot Generation
