import pickle

def save_txt(data, file_path):
    # Write the full summary to the text file
    with open(file_path, 'w') as f:
        f.write(data)

def save_sorted_df(path_to_file, sorted_dfs):
    with open(path_to_file, 'wb') as file:
        pickle.dump(sorted_dfs, file)
    print('File has been succesfully saved at: ' + path_to_file)

def load_sorted_df(path_to_file):
    with open(path_to_file, 'rb') as file:
        sorted_dfs = pickle.load(file)
    return sorted_dfs



