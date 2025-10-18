import cudf
import cupy
import matplotlib.pyplot as plt
from cuml.cluster import KMeans as cuKMeans
from cuml.datasets import make_blobs
from sklearn.cluster import KMeans as skKMeans
from sklearn.metrics import calinski_harabasz_score
from cuml import cuda
import os
import numpy as np
import time

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

def get_score(label_file):
    input_file = "input.txt"
    data = load_data(input_file)
    labels = load_labels(label_file)

    assert data.shape[0] == labels.shape[0], "Mismatch between points and labels."

    # score = silhouette_score(data, labels)
    score = calinski_harabasz_score(data, labels)
    return score


def do_benchmarking_cuml_n(file, start, end, step):
    n_values = []
    cuml_times = []
    my_times = []
    cuml_scores = []
    my_scores = []
    for n in range(start, end+1, step):
        n_values.append(n)
        d = 5
        k = 5


        os.system('python3 generator.py '+str(n)+' '+str(d)+' '+str(k)+' . input.txt')

        with open("input.txt", "r") as f:
            lines = f.readlines()
            host_data = []
            for line in lines[3:]:
                point = list(map(float, line.strip().split()))
                host_data.append(point)
        
        os.system('python3 Means_k_nvidia.py < input.txt > nvidia_labels.txt')

        with open("nvidia_timing.out", "r") as f:
            cuml_time = f.readline().split()[1]
        cuml_time = float(cuml_time)
        cuml_times.append(cuml_time)
        
        with open('nvidia_labels.txt', 'r') as f:
            cuml_labels = f.readline()

        cuml_score = get_score("nvidia_labels.txt")
        cuml_scores.append(cuml_score)

        
        os.system('nvcc -arch=sm_60 '+file+' -o benchmark')
        os.system('./benchmark < input.txt > my_labels.txt')
        my_labels = np.loadtxt("my_labels.txt")
        my_score = get_score("my_labels.txt")
        with open('cuda_timing.out', 'r') as f:
            my_time = f.readline().split()[1]
        my_time = float(my_time[:-3])
        my_times.append(my_time)
        my_scores.append(my_score)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot 1: Time comparison
    axs[0].plot(n_values, cuml_times, label='cuML KMeans')
    axs[0].plot(n_values, my_times, label='My KMeans')
    axs[0].set_xlabel('Number of points')
    axs[0].set_ylabel('Time (ms)')
    axs[0].set_title('Execution Time')
    axs[0].legend()

    # Plot 2: score comparison
    axs[1].plot(n_values, cuml_scores, label='cuML KMeans')
    axs[1].plot(n_values, my_scores, label='My KMeans')
    axs[1].set_xlabel('Number of points')
    axs[1].set_ylabel('Calinski Harabasz Score')
    axs[1].set_title('Clustering Quality')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def do_benchmarking_cuml_d(file):
    d_values = []
    cuml_times = []
    my_times = []
    cuml_scores = []
    my_scores = []
    for d in range(10, 101, 10):
        d_values.append(d)
        n = 100000
        k = 5


        os.system('python3 generator.py '+str(n)+' '+str(d)+' '+str(k)+' . input.txt')

        with open("input.txt", "r") as f:
            lines = f.readlines()
            host_data = []
            for line in lines[3:]:
                point = list(map(float, line.strip().split()))
                host_data.append(point)
        
        os.system('python3 Means_k_nvidia.py < input.txt > nvidia_labels.txt')

        with open("nvidia_timing.out", "r") as f:
            cuml_time = f.readline().split()[1]
        cuml_time = float(cuml_time)
        cuml_times.append(cuml_time)
        
        with open('nvidia_labels.txt', 'r') as f:
            cuml_labels = f.readline()

        cuml_score = get_score("nvidia_labels.txt")
        cuml_scores.append(cuml_score)

        
        os.system('nvcc -arch=sm_60 '+file+' -o benchmark')
        os.system('./benchmark < input.txt > my_labels.txt')
        my_labels = np.loadtxt("my_labels.txt")
        my_score = get_score("my_labels.txt")
        with open('cuda_timing.out', 'r') as f:
            my_time = f.readline().split()[1]
        my_time = float(my_time[:-3])
        my_times.append(my_time)
        my_scores.append(my_score)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot 1: Time comparison
    axs[0].plot(d_values, cuml_times, label='cuML KMeans')
    axs[0].plot(d_values, my_times, label='My KMeans')
    axs[0].set_xlabel('Number of points')
    axs[0].set_ylabel('Time (ms)')
    axs[0].set_title('Execution Time')
    axs[0].legend()

    # Plot 2: score comparison
    axs[1].plot(d_values, cuml_scores, label='cuML KMeans')
    axs[1].plot(d_values, my_scores, label='My KMeans')
    axs[1].set_xlabel('Number of points')
    axs[1].set_ylabel('Calinski Harabasz Score')
    axs[1].set_title('Clustering Quality')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def do_benchmarking_cuml_k(file):
    k_values = []
    cuml_times = []
    my_times = []
    cuml_scores = []
    my_scores = []
    for k in range(10, 101, 10):
        k_values.append(k)
        n = 100000
        d = 5


        os.system('python3 generator.py '+str(n)+' '+str(d)+' '+str(k)+' . input.txt')

        with open("input.txt", "r") as f:
            lines = f.readlines()
            host_data = []
            for line in lines[3:]:
                point = list(map(float, line.strip().split()))
                host_data.append(point)
        
        os.system('python3 Means_k_nvidia.py < input.txt > nvidia_labels.txt')

        with open("nvidia_timing.out", "r") as f:
            cuml_time = f.readline().split()[1]
        cuml_time = float(cuml_time)
        cuml_times.append(cuml_time)
        
        with open('nvidia_labels.txt', 'r') as f:
            cuml_labels = f.readline()

        cuml_score = get_score("nvidia_labels.txt")
        cuml_scores.append(cuml_score)

        
        os.system('nvcc -arch=sm_60 '+file+' -o benchmark')
        os.system('./benchmark < input.txt > my_labels.txt')
        my_labels = np.loadtxt("my_labels.txt")
        my_score = get_score("my_labels.txt")
        with open('cuda_timing.out', 'r') as f:
            my_time = f.readline().split()[1]
        my_time = float(my_time[:-3])
        my_times.append(my_time)
        my_scores.append(my_score)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot 1: Time comparison
    axs[0].plot(k_values, cuml_times, label='cuML KMeans')
    axs[0].plot(k_values, my_times, label='My KMeans')
    axs[0].set_xlabel('Number of points')
    axs[0].set_ylabel('Time (ms)')
    axs[0].set_title('Execution Time')
    axs[0].legend()

    # Plot 2: score comparison
    axs[1].plot(k_values, cuml_scores, label='cuML KMeans')
    axs[1].plot(k_values, my_scores, label='My KMeans')
    axs[1].set_xlabel('Number of points')
    axs[1].set_ylabel('Calinski Harabasz Score')
    axs[1].set_title('Clustering Quality')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def do_benchmarking_all_n(d, k):
    files = []
    for file in os.listdir('All_implementations'):
        if file.endswith(".cu"):
            files.append(file)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
    
    n_values = []
    times = dict()
    scores = dict()
    for n in range(10000, 100001, 10000):
        n_values.append(n)


        os.system('python3 generator.py '+str(n)+' '+str(d)+' '+str(k)+' . input.txt')

        with open("input.txt", "r") as f:
            lines = f.readlines()
            host_data = []
            for line in lines[3:]:
                point = list(map(float, line.strip().split()))
                host_data.append(point)
        
        for file in files:
            os.system('nvcc -arch=sm_60 All_implementations/'+file +' -o benchmark')
            os.system('./benchmark < input.txt > labels.txt')

            with open('cuda_timing.out', "r") as f:
                time = f.readline().split()[1]
            time = float(time[:-3])
            times[file] = times.get(file, []) + [time]

            score = get_score("labels.txt")
            scores[file] = scores.get(file, []) + [score]

    for file in files:
        # Plot 1: Time comparison
        axs[0].plot(n_values, times[file], label=f'KMeans {file}')
        axs[0].set_xlabel('Number of points')
        axs[0].set_ylabel('Time (ms)')
        axs[0].set_title('Execution Time')
        axs[0].legend()

        # Plot 2: score comparison
        axs[1].plot(n_values, scores[file], label=f'KMeans{file}')
        axs[1].set_xlabel('Number of points')
        axs[1].set_ylabel('Calinski Harabasz Score')
        axs[1].set_title('Clustering Quality')
        axs[1].legend()
    plt.tight_layout()
    plt.show()



def do_benchmarking_all_d(n, k):
    files = []
    for file in os.listdir('All_implementations'):
        if file.endswith(".cu"):
            files.append(file)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
    
    d_values = []
    times = dict()
    scores = dict()
    for d in range(10, 101, 10):
        d_values.append(d)


        os.system('python3 generator.py '+str(n)+' '+str(d)+' '+str(k)+' . input.txt')

        with open("input.txt", "r") as f:
            lines = f.readlines()
            host_data = []
            for line in lines[3:]:
                point = list(map(float, line.strip().split()))
                host_data.append(point)
        
        for file in files:
            os.system('nvcc -arch=sm_60 All_implementations/'+file +' -o benchmark')
            os.system('./benchmark < input.txt > labels.txt')

            with open('cuda_timing.out', "r") as f:
                time = f.readline().split()[1]
            time = float(time[:-3])
            times[file] = times.get(file, []) + [time]

            score = get_score("labels.txt")
            scores[file] = scores.get(file, []) + [score]

    for file in files:
        # Plot 1: Time comparison
        axs[0].plot(d_values, times[file], label=f'KMeans {file}')
        axs[0].set_xlabel('Dimensions')
        axs[0].set_ylabel('Time (ms)')
        axs[0].set_title('Execution Time')
        axs[0].legend()

        # Plot 2: score comparison
        axs[1].plot(d_values, scores[file], label=f'KMeans{file}')
        axs[1].set_xlabel('Dimensions')
        axs[1].set_ylabel('Calinski Harabasz Score')
        axs[1].set_title('Clustering Quality')
        axs[1].legend()
    plt.tight_layout()
    plt.show()



def do_benchmarking_all_k(n, d):
    files = []
    for file in os.listdir('All_implementations'):
        if file.endswith(".cu"):
            files.append(file)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
    
    k_values = []
    times = dict()
    scores = dict()
    for k in range(10, 101, 10):
        k_values.append(k)


        os.system('python3 generator.py '+str(n)+' '+str(d)+' '+str(k)+' . input.txt')

        with open("input.txt", "r") as f:
            lines = f.readlines()
            host_data = []
            for line in lines[3:]:
                point = list(map(float, line.strip().split()))
                host_data.append(point)
        
        for file in files:
            os.system('nvcc -arch=sm_60 All_implementations/'+file +' -o benchmark')
            os.system('./benchmark < input.txt > labels.txt')

            with open('cuda_timing.out', "r") as f:
                time = f.readline().split()[1]
            time = float(time[:-3])
            times[file] = times.get(file, []) + [time]

            score = get_score("labels.txt")
            scores[file] = scores.get(file, []) + [score]

    for file in files:
        # Plot 1: Time comparison
        axs[0].plot(k_values, times[file], label=f'KMeans {file}')
        axs[0].set_xlabel('Centroids')
        axs[0].set_ylabel('Time (ms)')
        axs[0].set_title('Execution Time')
        axs[0].legend()

        # Plot 2: score comparison
        axs[1].plot(k_values, scores[file], label=f'KMeans{file}')
        axs[1].set_xlabel('Centroids')
        axs[1].set_ylabel('Calinski Harabasz Score')
        axs[1].set_title('Clustering Quality')
        axs[1].legend()
    plt.tight_layout()
    plt.show()



do_benchmarking_all_n(5, 5)
do_benchmarking_all_d(100000, 5)
do_benchmarking_all_k(100000, 5)