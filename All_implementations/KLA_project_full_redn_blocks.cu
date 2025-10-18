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
#define MAX_SIZE 1e10
int NUM_PARTITIONS;


#define assign_centroid(device_x, N, d, k, dist_pref_sum, device_centroids, device_new_nearest_centroid, device_nearest_centroid, device_change, device_count, dist_cents, stream) \
    int NUM_BLOCKS = (N*d*k + NUM_THREADS_PTS - 1) / NUM_THREADS_PTS; \
    make_dist<<<NUM_BLOCKS, NUM_THREADS_PTS>>>(device_x, N, d, k, device_centroids, dist_pref_sum); \
    make_dist_sum<<<k*N, d, d * sizeof(double)>>>(N, d, k, dist_pref_sum); \
    NUM_BLOCKS = (N*k + NUM_THREADS_PTS - 1) / NUM_THREADS_PTS; \
    assign_dist_cent<<<NUM_BLOCKS, NUM_THREADS_PTS>>>(N, k, d, dist_pref_sum, dist_cents); \
    find_centroid<<<NUM_BLOCKS_PTS, NUM_THREADS_PTS>>>(N, k, d, dist_cents, device_nearest_centroid, device_new_nearest_centroid, device_change, device_count);


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
        // count_pfx[idx] += temp_int[idx - off];
        // printf("count_pfx[%d] = %d\n", idx, count_pfx[idx]);
        // printf("temp_int[%d] = %d\n", idx - off, temp_int[idx - off]);
        int val = idx & ((off << 1) - 1);
        if (val >= off) {
            // printf("count_pfx[%d] += count_pfx[%d] :: (%d += %d)\n", idx, idx - val + off - 1, count_pfx[idx], count_pfx[idx - val + off - 1]);
            count_pfx[idx] += count_pfx[idx - val + off - 1];
        }
    }
}

__global__ void do_partial_prefix_sum_double(double *centroid_acc, int n, int d, int off) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * d) return;
    int i = idx / d;
    int j = idx % d;
    // if(i >= off) {
    //     centroid_acc[idx] += temp_double[(i - off) * d + j];
    // }
    if (i >= off) {
        int val = (i & ((off << 1) - 1));
        if (val >= off) {
            // printf("centroid_acc[%d][%d] += centroid_acc[%d][%d] :: (%lf += %lf)\n", i, j, i - val + off - 1, j, centroid_acc[i * d + j], centroid_acc[(i - val + off - 1) * d + j]);
            centroid_acc[idx] += centroid_acc[(i - val + off - 1) * d + j];
        }
    }
}

__global__ void print_array(double *device_array, int len) {
    for (int i = 0; i < len; i++) {
        printf("%lf ", device_array[i]);
    }
    printf("\n");
}


// launch on threads = N*d*k
__global__ void make_dist(int *x, int N, int d, int k, double *device_centroids, double *dist_pref_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*d*k) return;
    int k0 = idx / (N*d);
    int idx1 = idx % (N*d);
    int d0 = idx1 / N;
    int n0 = idx1 % N;
    double diff = (x(n0, d0) - device_centroids(k0, d0));

    dist_pref_sum[idx] = diff * diff;

    /*
    device_new_nearest_centroid[idx] = mn_dist_cent;
    int old_centroid = device_nearest_centroid[idx];
    if (old_centroid != mn_dist_cent) {
        //printf("Change: %d to %d for %d\n", old_centroid, mn_dNUM_PARTITIONSist_cent, idx);
        *device_change = 1;
        atomicAdd(&count[mn_dist_cent], 1);
        atomicAdd(&count[old_centroid], -1);
    }
    */
}

// launch on threads = N*d*k
__global__ void make_dist_sum(int N, int d, int k, double *dist_pref_sum){
    extern __shared__ double dist_pref_shared[];
    int tid = threadIdx.x;
    int k0 = blockIdx.x/N;
    int n0 = blockIdx.x%N;
    dist_pref_shared[tid] = dist_pref_sum[k0*N*d + tid*N + n0];
    __syncthreads();

    for(int off=1; off < 2*d; off*=2){
        if(tid >= off){
            int val = tid & (2*off-1);
            if(val >= off){
                dist_pref_shared[tid] += dist_pref_shared[tid - val + off - 1];
            }
        }
        __syncthreads();
    }
    dist_pref_sum[k0*N*d + (d-1)*N + n0] = dist_pref_shared[d-1];
}

// launch on threads = N*k
__global__ void assign_dist_cent(int N, int k, int d, double *dist_pref_sum, double *dist_cents){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*k) return;
    int k0 = idx / N;
    int n0 = idx % N;
    dist_cents[idx] = dist_pref_sum[k0*N*d + (d-1)*N + n0];
}

// launch on threads = N
__global__ void find_centroid(int N, int k, int d, double *dist_cent, int *device_nearest_centroid, int *device_new_nearest_centroid, int *device_change, int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    double mn_dist_cent = DBL_MAX;
    int mn_dist_idx = -1;
    for(int i=0;i<k;i++){
        if(dist_cent[i*N + idx] < mn_dist_cent){
            mn_dist_cent = dist_cent[i*N + idx];
            mn_dist_idx = i;
        }
    }
    device_new_nearest_centroid[idx] = mn_dist_idx;
    int old_centroid = device_nearest_centroid[idx];
    if (old_centroid != mn_dist_idx) {
        *device_change = 1;
        device_nearest_centroid[idx] = mn_dist_idx;
        atomicAdd(&count[mn_dist_idx], 1);
        atomicAdd(&count[old_centroid], -1);
    }
}


// launch on threads = number of points
__global__ void update_k_means_1(double *device_centroids_acc, int *x, int N, int d, int k, int *device_nearest_centroid, int *count_prefix, int *centroid_start_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int i = device_nearest_centroid[idx];
    // for (int j = 0; j < d; j++) {
    //     atomicAdd(&device_centroids(i, j), x(idx, j));
    // }
    int off = atomicInc((unsigned int *)&centroid_start_idx[i], INT_MAX);

    int ii = (i - 1) >= 0 ? count_prefix[i - 1] : 0;

    for (int j = 0; j < d; j++) {
        device_centroids_acc((ii + off), j) = x(idx, j);
        // printf("device_centroids_acc[%d][%d] = [%d] = %f\n", ii+off, j, (ii+off)*d + j,device_centroids_acc((ii+off), j));
        // printf("x[%d][%d] = %f\n", idx, j, x(idx, j));
    }
}

// cuML implementation of k-means cross check

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
    int *x = (int *)malloc(d * N * sizeof(int));
    double *centroids = (double *)malloc(d * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            cin >> x(i, j);
            if (i < k) {
                centroids[i + j * k] = x(i, j);
            }
        }
    }

    NUM_PARTITIONS = MAX_SIZE / (N * d);
    if(NUM_PARTITIONS > k) {
        NUM_PARTITIONS = k;
    }
    // start time
    // cout << "NUM_PARTITIONS: " << NUM_PARTITIONS << endl;
    auto start_total = std::chrono::high_resolution_clock::now();

    int *device_x;
    double *device_centroids;
    double *dist_pref_sum, *dist_cents;
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
    cudaMalloc(&dist_cents, N * k * sizeof(double));

    cudaMalloc(&dist_pref_sum, NUM_PARTITIONS * N * d * sizeof(double));

    cudaMalloc(&temp_int, k * sizeof(int));
    cudaMalloc(&temp_double, N * d * sizeof(double));

    cudaMalloc(&device_x, d * N * sizeof(int));
    cudaMalloc(&device_centroids, d * k * sizeof(double));
    cudaMalloc(&device_nearest_centroid, N * sizeof(int));
    cudaMalloc(&device_new_nearest_centroid, N * sizeof(int));
    cudaMalloc(&device_change, sizeof(int));
    cudaMalloc(&device_count, k * sizeof(int));
    cudaMalloc(&device_count_prefix, k * sizeof(int));
    cudaMalloc(&device_centroids_acc, N * d * sizeof(double));

    cudaMemset(device_nearest_centroid, 0, N * sizeof(int));

    cudaMemcpy(device_x, x, d * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_centroids, centroids, k * d * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemset(device_count, 0, k * sizeof(int));
    cudaMemcpy(device_count, &N, sizeof(int), cudaMemcpyHostToDevice);

    // cudaMemcpy(count, device_count, k * sizeof(int), cudaMemcpyDeviceToHost);
    // cout << "Count: ";
    // for(int i=0;i<k;i++){
    //     cout << count[i] << " ";
    // }
    // cout << endl;

    auto start_calc = std::chrono::high_resolution_clock::now();

    int NUM_BLOCKS_PTS = (N + NUM_THREADS_PTS - 1) / NUM_THREADS_PTS;
    int NUM_BLOCKS_CENT = (k + NUM_THREADS_CENT - 1) / NUM_THREADS_CENT;
    cudaError_t err;

    cudaMemset(device_change, 0, sizeof(int));
    assign_centroid(device_x, N, d, k, dist_pref_sum, device_centroids, device_new_nearest_centroid, device_nearest_centroid, device_change, device_count, dist_cents, stream);
    cudaMemcpy(&change, device_change, sizeof(int), cudaMemcpyDeviceToHost);


    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed 0: %s\n", cudaGetErrorString(err));
    }
    // cudaMemcpy(centroids, device_centroids, k * d * sizeof(double), cudaMemcpyDeviceToHost);
    // cout << "Centroids: ";
    // for(int i=0;i<k;i++){
    //     for(int j=0;j<d;j++){
    //         cout << centroids[i+j*k] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;
    // double *centroid_acc = (double *)malloc(N * d * sizeof(double));
    while (change) {
        // cudaMemcpy(device_nearest_centroid, device_new_nearest_centroid, N * sizeof(int), cudaMemcpyDeviceToDevice);
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

        // cout << "Count New: ";
        // for(int i=0;i<k;i++){
        //     cout << count_prefix[i] << " ";
        // }

        cudaMemset(temp_int, 0, k * sizeof(int));

        update_k_means_1<<<NUM_BLOCKS_PTS, NUM_THREADS_PTS>>>(device_centroids_acc, device_x, N, d, k, device_nearest_centroid, device_count_prefix, temp_int);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed 3: %s\n", cudaGetErrorString(err));
        }

        // cudaMemcpy(centroid_acc, device_centroids_acc, N * d * sizeof(double), cudaMemcpyDeviceToHost);
        // cout << "Centroid Acc before: \n";
        // for(int i=0;i<N;i++){
        //     for(int j=0;j<d;j++){
        //         cout << centroid_acc[i*d+j] << " ";
        //     }
        //     cout << endl;
        // }
        // cout << endl;

        for (int i = 0; i < k; i++) {
            // cout << "i: " << i << endl;
            int curr = ((i - 1) >= 0 ? count_prefix[i - 1] : 0);
            int n = (count_prefix[i] - curr);
            PREFIX_SUM_DOUBLE((device_centroids_acc + curr * d), temp_double, n, d, stream[i]);
            // err = cudaGetLastError();
            // if (err != cudaSuccess) {
            //     printf("Kernel launch failed 4: %s\n", cudaGetErrorString(err));
            // }
        }

        cudaDeviceSynchronize();

        // cudaMemcpy(centroid_acc, device_centroids_acc, N * d * sizeof(double), cudaMemcpyDeviceToHost);
        // cout << "Centroid Acc after: \n";
        // for(int i=0;i<N;i++){
        //     for(int j=0;j<d;j++){
        //         cout << centroid_acc[i*d+j] << " ";
        //     }
        //     cout << endl;
        // }
        // cout << endl;

        update_k_means_2<<<NUM_BLOCKS_CENT, NUM_THREADS_CENT>>>(device_centroids_acc, device_centroids, N, d, k, device_count_prefix, device_count);
        // err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     printf("Kernel launch failed 5: %s\n", cudaGetErrorString(err));
        // }

        cudaMemset(device_change, 0, sizeof(int));

        assign_centroid(device_x, N, d, k, dist_pref_sum, device_centroids, device_new_nearest_centroid, device_nearest_centroid, device_change, device_count, dist_cents, stream);
        // err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     printf("Kernel launch failed 6: %s\n", cudaGetErrorString(err));
        // }

        cudaMemcpy(centroids, device_centroids, k * d * sizeof(double), cudaMemcpyDeviceToHost);
        // cout << "Centroids: ";
        // for(int i=0;i<k;i++){
        //     for(int j=0;j<d;j++){
        //         cout << centroids[i+j*k] << " ";
        //     }
        //     cout << endl;
        // }
        // cout << endl;

        cudaMemcpy(&change, device_change, sizeof(int), cudaMemcpyDeviceToHost);

        // cudaMemcpy(y, device_nearest_centroid, N * sizeof(int), cudaMemcpyDeviceToHost);
        // cout << "Nearest Centroid: \n";
        // for(int i=0;i<N;i++){
        //     cout << y[i] << " ";
        // }
        // cout << endl;

        // cout << "Change: " << change << endl;
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