import pandas as pd

# Load raw data
raw_data = pd.read_sas(
            "data\CCHS Annual 2015_2016 PUMF\data_donnee\hs.sas7bdat")

# Data columns that are of interest
column_labels = {'GEO_PRV': 'Province',
                 'GEODGHR4': 'Health Region',
                 'DHH_SEX': 'Gender',
                 'DHHGAGE': 'Age',
                 'MAC_010': 'Employment Status',
                 'MAC_015': 'Attending School?',
                 'HWTDGCOR': 'BMI (adjusted)',
                 'SDCDGCGT': 'Cultural/Racial Background'}

# Extract the data columns of interest, rename them and filter them
# for adjusted BMI outliers
max_adjusted_BMI = 100

desired_columns = raw_data[column_labels.keys()]
desired_columns = desired_columns.rename(columns = column_labels)
desired_columns = desired_columns[
    desired_columns['BMI (adjusted)'] < max_adjusted_BMI]


# Save the processed data
desired_columns.to_pickle('data\processed_data.csv')
