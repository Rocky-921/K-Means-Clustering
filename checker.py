import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import os
import sys

def load_data(input_file):
    with open(input_file, 'r') as f:
        N = int(f.readline())
        d = int(f.readline())
        k = int(f.readline())
        data = []
        for _ in range(N):
            point = list(map(int, f.readline().split()))
            data.append(point)
    return np.array(data)

def load_labels(label_file):
    return np.loadtxt(label_file, dtype=int)

def score(label_file, input_file):
    data = load_data(input_file)
    labels = load_labels(label_file)

    assert data.shape[0] == labels.shape[0], "Mismatch between points and labels."

    # score = silhouette_score(data, labels)
    score = calinski_harabasz_score(data, labels)
    return score


input_dir = "input"
my_output_dir= "my_output"
inbuilt_output_dir = "inbuilt_output"

# for all files in input directory

for file in os.listdir('All_implementations'):
    if not file.endswith(".cu"):
        continue
    os.system(f"nvcc -arch=sm_60 ./All_implementations/{file} -o KLA_project")
    
    logger = "logs/logfile" + file[11:-3]

    os.system("mkdir -p logs")
    os.system(f"mkdir -p {my_output_dir}")
    os.system(f"mkdir -p {inbuilt_output_dir}")
    os.system(f"> {logger}")

    files = os.listdir(input_dir)
    files = [int(f) for f in files]
    files.sort()
    files = [str(f) for f in files]

    for file in files:
        input_file = os.path.join(input_dir, file)
        my_output_file = os.path.join(my_output_dir, file)
        os.system(f"./KLA_project < {input_file} > {my_output_file}")

        inbuilt_output_file = os.path.join(inbuilt_output_dir, file)
        os.system(f"python Means_k_nvidia.py < {input_file} > {inbuilt_output_file}")

        my_score = score(my_output_file, input_file)
        inbuilt_score = score(inbuilt_output_file, input_file)
        print("My Score: ", my_score)
        print("Inbuilt Score: ", inbuilt_score)
        print("Difference: ", abs(my_score - inbuilt_score))
        print("--------------------------------------------------")

        with open('cuda_timing.out', 'r') as f:
            my_time = f.readline()
        
        with open('nvidia_timing.out', 'r') as f:
            inbuilt_time = f.readline()

        with open(logger, 'a') as f:
            f.write(f"File: {file}\n")
            f.write(f"My Score: {my_score}\n")
            f.write(f"Inbuilt Score: {inbuilt_score}\n")
            f.write(f"Difference: {(my_score - inbuilt_score)}\n")
            f.write(f"My Time: {my_time}\n")
            f.write(f"Nvidia Time: {inbuilt_time}\n")
            f.write("--------------------------------------------------\n")

    with open(logger, 'a') as f:
        f.write("All files processed.\n")
        f.write("--------------------------------------------------\n")
        f.write("CUDA program executed successfully.\n")
        f.write("--------------------------------------------------\n")