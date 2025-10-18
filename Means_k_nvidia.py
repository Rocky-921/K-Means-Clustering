import sys
import numpy as np
import cupy as cp
import cudf
from cuml.cluster import KMeans as cuKMeans
import time

def main():
    # Read input from standard input
    N = int(sys.stdin.readline())     # number of points
    d = int(sys.stdin.readline())     # dimensions
    k = int(sys.stdin.readline())     # number of clusters

    data = []
    for _ in range(N):
        point = list(map(float, sys.stdin.readline().split()))
        data.append(point)

    # Convert to NumPy array (host-side)
    host_data = np.array(data, dtype=np.float32)

    # Convert to CuPy array (GPU device-side)
    start_time = time.time()
    device_data = cp.asarray(host_data)
    # Initialize cuML KMeans
    kmeans_cuml = cuKMeans(init="k-means||", n_clusters=k, random_state=0)
    # Time the KMeans fitting process
    kmeans_cuml.fit(device_data)
    labels = kmeans_cuml.labels_.get()
    end_time = time.time()

    # Get labels from device to host

    # Output cluster labels to standard output
    for label in labels:
        print(int(label), end=' ')
    print()

    # Calculate elapsed time for cuML KMeans fitting
    elapsed_time = (end_time - start_time)*1000  # Convert to milliseconds

    # Store the elapsed time in an output file
    with open('nvidia_timing.out', 'w') as f:
        f.write(f"Total: {elapsed_time:.4f} ms\n")

if __name__ == "__main__":
    main()
