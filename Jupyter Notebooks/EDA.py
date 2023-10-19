import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Auxiliary Functions

def save_txt(data, file_path):
    # Write the full summary to the text file
    with open(file_path, 'w') as f:
        f.write(data)


path_to_data = '/Users/thomasgaehtgens/Documents/Graduate Quants Preparation/OPTIVER /KAGGLE/OPTIVER_Trading_at_the_Close/data/train.csv'
df = pd.read_csv(path_to_data)

# get summary statistics
full_summary = df.describe().to_string()


from IPython import embed; embed()
