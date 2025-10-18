#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#include <chrono>
#include <fstream>
#include <iostream>

using namespace std;

#define shared_centroids(i, j) shared_centroids[i + j * k]
#define device_centroids(i, j) device_centroids[i + j * k]
#define device_centroids_acc(i, j) device_centroids_acc[i * d + j]
#define new_centroid(i, j) new_centroid[i + j * k]
#define x(i, j) x[i + j * N]

// tune parameters
#define NUM_THREADS_PTS 512
#define NUM_THREADS_CENT 512

#define PREFIX_SUM_INT(count_pfx, k)                                                         \
    for (int off = 1; off < 2 * k; off *= 2) {                                               \
        do_partial_prefix_sum_int<<<NUM_BLOCKS_CENT, NUM_THREADS_CENT>>>(count_pfx, k, off); \
    }

#define PREFIX_SUM_DOUBLE(device_centroid_acc, temp, n, d, stream)                                       \
    for (int off = 1; off < 2 * n && n > 0; off *= 2) {                                                  \
        int NUM_BLOCKS__ = (n * d + NUM_THREADS_PTS - 1) / NUM_THREADS_PTS;                              \
        do_partial_prefix_sum_double<<<NUM_BLOCKS__, NUM_THREADS_PTS, 0, stream>>>(device_centroid_acc, n, d, off); \
    }

__global__ void do_partial_prefix_sum_int(int *count_pfx, int k, int off) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    if (idx >= off) {
        int val = idx & ((off << 1) - 1);
        if (val >= off) {
            count_pfx[idx] += count_pfx[idx - val + off - 1];
        }
    }
}

__global__ void do_partial_prefix_sum_double(double *centroid_acc, int n, int d, int off) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * d) return;
    int i = idx / d;
    int j = idx % d;
    if (i >= off) {
        int val = (i & ((off << 1) - 1));
        if (val >= off) {
            centroid_acc[idx] += centroid_acc[(i - val + off - 1) * d + j];
        }
    }
}

__device__ void print_nearest_centroid(int *device_nearest_centroid, int N) {
    for (int i = 0; i < N; i++) {
        printf("%d ", device_nearest_centroid[i]);
    }
    printf("\n");
}

// launch on threads = number of points
__global__ void assign_centroid(double *x, int N, int d, int k, double *device_centroids, int *device_new_nearest_centroid, int *device_nearest_centroid, int *device_change, int *count) {
    extern __shared__ float shared_centroids[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int shared_len = k * d;
    for (int i = threadIdx.x; i < shared_len; i += blockDim.x) {
        shared_centroids[i] = device_centroids[i];
    }
    __syncthreads();
    if (idx >= N) return;

    double x_local[100];
    for (int j = 0; j < d; j++) {
        x_local[j] = x(idx, j);  // Coalesced global read
    }
    double mn_dist = DBL_MAX;
    int mn_dist_cent = -1;
    for (int i = 0; i < k; i++) {
        double curr_dist = 0;
        for (int j = 0; j < d; j++) {
            double diff = (x_local[j] - shared_centroids(i, j));
            curr_dist += diff * diff;
        }
        if (curr_dist < mn_dist) {
            mn_dist = curr_dist;
            mn_dist_cent = i;
        }
    }
    device_new_nearest_centroid[idx] = mn_dist_cent;
    int old_centroid = device_nearest_centroid[idx];
    if (old_centroid != mn_dist_cent) {
        *device_change = 1;
        atomicAdd(&count[mn_dist_cent], 1);
        atomicAdd(&count[old_centroid], -1);
    }
}

// launch on threads = number of points
__global__ void update_k_means_1(double *device_centroids_acc, double *x, int N, int d, int k, int *device_nearest_centroid, int *count_prefix, int *centroid_start_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int i = device_nearest_centroid[idx];

    int off = atomicInc((unsigned int *)&centroid_start_idx[i], INT_MAX);

    int ii = (i - 1) >= 0 ? count_prefix[i - 1] : 0;

    for (int j = 0; j < d; j++) {
        device_centroids_acc((ii + off), j) = x(idx, j);
    }
}


// launch on threads = number of centroids
__global__ void update_k_means_2(double *device_centroids_acc, double *device_centroids, int N, int d, int k, int *count_prefix, int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    int end_idx = count_prefix[idx] - 1;
    for (int j = 0; j < d; j++) {
        device_centroids(idx, j) = device_centroids_acc(end_idx, j) / count[idx];
    }
}
int main() {
    int N;
    int d;
    int k;
    cin >> N;
    cin >> d;
    cin >> k;
    double *x = (double *)malloc(d * N * sizeof(double));
    double *centroids = (double *)malloc(d * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            cin >> x(i, j);
            if (i < k) {
                centroids[i + j * k] = x(i, j);
            }
        }
    }

    // start time
    auto start_total = std::chrono::high_resolution_clock::now();

    double *device_x;
    double *device_centroids;
    int *device_nearest_centroid, *device_new_nearest_centroid;
    int *y = (int *)malloc(N * sizeof(int));
    int *device_change, change;
    int *device_count, *device_count_prefix, *count_prefix = (int *)malloc(k * sizeof(int));
    int *temp_int;
    double *temp_double;
    double *device_centroids_acc;
    int *count = (int *)malloc(k * sizeof(int));

    cudaStream_t stream[k];
    stream[0] = 0;
    for (int i = 1; i < k; i++) {
        cudaStreamCreate(&stream[i]);
    }

    cudaMalloc(&temp_int, k * sizeof(int));
    cudaMalloc(&temp_double, N * d * sizeof(double));

    cudaMalloc(&device_x, d * N * sizeof(double));
    cudaMalloc(&device_centroids, d * k * sizeof(double));
    cudaMalloc(&device_nearest_centroid, N * sizeof(int));
    cudaMalloc(&device_new_nearest_centroid, N * sizeof(int));
    cudaMalloc(&device_change, sizeof(int));
    cudaMalloc(&device_count, k * sizeof(int));
    cudaMalloc(&device_count_prefix, k * sizeof(int));
    cudaMalloc(&device_centroids_acc, N * d * sizeof(double));

    cudaMemset(device_nearest_centroid, 0, N * sizeof(int));

    cudaMemcpy(device_x, x, d * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_centroids, centroids, k * d * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemset(device_count, 0, k * sizeof(int));
    cudaMemcpy(device_count, &N, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(count, device_count, k * sizeof(int), cudaMemcpyDeviceToHost);


    auto start_calc = std::chrono::high_resolution_clock::now();

    int NUM_BLOCKS_PTS = (N + NUM_THREADS_PTS - 1) / NUM_THREADS_PTS;
    int NUM_BLOCKS_CENT = (k + NUM_THREADS_CENT - 1) / NUM_THREADS_CENT;
    cudaError_t err;

    cudaMemset(device_change, 0, sizeof(int));
    assign_centroid<<<NUM_BLOCKS_PTS, NUM_THREADS_PTS, k * d * sizeof(float)>>>(device_x, N, d, k, device_centroids, device_new_nearest_centroid, device_nearest_centroid, device_change, device_count);
    cudaMemcpy(&change, device_change, sizeof(int), cudaMemcpyDeviceToHost);

    // cudaMemcpy(centroids, device_centroids, k * d * sizeof(double), cudaMemcpyDeviceToHost);

    while (change) {
        cudaMemcpy(device_nearest_centroid, device_new_nearest_centroid, N * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(device_centroids, 0, d * k * sizeof(double));
        cudaMemcpy(device_count_prefix, device_count, k * sizeof(int), cudaMemcpyDeviceToDevice);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed 1: %s\n", cudaGetErrorString(err));
        }

        PREFIX_SUM_INT(device_count_prefix, k);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed 2: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(count_prefix, device_count_prefix, k * sizeof(int), cudaMemcpyDeviceToHost);



        cudaMemset(temp_int, 0, k * sizeof(int));

        update_k_means_1<<<NUM_BLOCKS_PTS, NUM_THREADS_PTS>>>(device_centroids_acc, device_x, N, d, k, device_nearest_centroid, device_count_prefix, temp_int);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed 3: %s\n", cudaGetErrorString(err));
        }


        for (int i = 0; i < k; i++) {
            int curr = ((i - 1) >= 0 ? count_prefix[i - 1] : 0);
            int n = (count_prefix[i] - curr);
            PREFIX_SUM_DOUBLE((device_centroids_acc + curr * d), temp_double, n, d, stream[i]);

        }

        cudaDeviceSynchronize();



        update_k_means_2<<<NUM_BLOCKS_CENT, NUM_THREADS_CENT>>>(device_centroids_acc, device_centroids, N, d, k, device_count_prefix, device_count);


        cudaMemset(device_change, 0, sizeof(int));

        assign_centroid<<<NUM_BLOCKS_PTS, NUM_THREADS_PTS, k * d * sizeof(float)>>>(device_x, N, d, k, device_centroids, device_new_nearest_centroid, device_nearest_centroid, device_change, device_count);


        // cudaMemcpy(centroids, device_centroids, k * d * sizeof(double), cudaMemcpyDeviceToHost);


        cudaMemcpy(&change, device_change, sizeof(int), cudaMemcpyDeviceToHost);


    }

    auto end_calc = std::chrono::high_resolution_clock::now();

    cudaMemcpy(y, device_nearest_centroid, N * sizeof(int), cudaMemcpyDeviceToHost);

    // end time
    auto end_total = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_total = end_total - start_total;
    std::chrono::duration<double> elapsed_calc = end_calc - start_calc;
    ofstream out("cuda_timing.out");
    out << "Total: " << elapsed_total.count() * 1000 << "ms, Calculation Time: " << elapsed_calc.count() * 1000 << "ms";
    out.close();
    for (int i = 0; i < N; i++) {
        cout << y[i] << " ";
    }
    cout << endl;

    cudaFree(device_x);
    cudaFree(device_centroids);
    cudaFree(device_nearest_centroid);
    cudaFree(device_new_nearest_centroid);
    cudaFree(device_change);
    cudaFree(device_count);
    free(x);
    free(y);
    return 0;
}