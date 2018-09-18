import os
import numpy as np

def generate_txt_files(path_to_toy_data):
    files = os.listdir(path_to_toy_data)
    train = open("trainval.txt", "w")

    size_train_set = int(len(files))

    train_files = np.random.choice(files, size=size_train_set, replace=False)
    for f in train_files:
        train.write(f.replace(".jpg", "") + " " + str(1) + "\n")
        files.remove(f)
    train.close()

    print(len(files))


# run in folder tensorflow_toy_detector/
generate_txt_files("images/test")