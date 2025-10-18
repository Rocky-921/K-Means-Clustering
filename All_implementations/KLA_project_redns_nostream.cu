#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#include <chrono>
#include <fstream>
#include <iostream>

using namespace std;

#define shared_centroids(i, j) shared_centroids[i + j * k]
#define device_centroids(i, j) device_centroids[i + j * k]
#define device_centroids_acc(i, j) device_centroids_acc[i*d + j]
#define new_centroid(i, j) new_centroid[i + j * k]
#define x(i, j) x[i + j * N]

// tune parameters
#define NUM_THREADS_PTS 512
#define NUM_THREADS_CENT 512

#define PREFIX_SUM_INT(count_pfx, temp_int, k) \
    for (int off = 1; off < k; off*=2) \
    { \
        cudaMemcpy(temp_int, count_pfx, k * sizeof(int), cudaMemcpyDeviceToDevice); \
        do_partial_prefix_sum_int<<<NUM_BLOCKS_CENT, NUM_THREADS_CENT>>>(count_pfx, temp_int, k, off); \
    }

#define PREFIX_SUM_DOUBLE(centroid_acc, temp_double, n, d) \
    for (int off = 1; off < n && n>0; off*=2) \
    { \
        cudaMemcpy(temp_double, centroid_acc, n * d * sizeof(double), cudaMemcpyDeviceToDevice); \
        int NUM_BLOCKS__ = (n*d + NUM_THREADS_PTS - 1) / NUM_THREADS_PTS; \
        do_partial_prefix_sum_double<<<NUM_BLOCKS__, NUM_THREADS_PTS>>>(centroid_acc, temp_double, n, d, off); \
    }

__global__ void do_partial_prefix_sum_int(int *count_pfx, int *temp_int, int k, int off){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= k) return;
    if(idx >= off) {
        count_pfx[idx] += temp_int[idx - off];
        //printf("count_pfx[%d] = %d\n", idx, count_pfx[idx]);
        //printf("temp_int[%d] = %d\n", idx - off, temp_int[idx - off]);
    }
}

__global__ void do_partial_prefix_sum_double(double *centroid_acc, double *temp_double, int n, int d, int off){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n*d) return;
    int i = idx / d;
    int j = idx % d;
    if(i >= off) {
        centroid_acc[idx] += temp_double[(i - off) * d + j];
    }
}

__device__ void print_nearest_centroid(int *device_nearest_centroid, int N) {
    for (int i = 0; i < N; i++) {
        printf("%d ", device_nearest_centroid[i]);
    }
    printf("\n");
}

// launch on threads = number of points
__global__ void assign_centroid(int *x, int N, int d, int k, double *device_centroids, int *device_new_nearest_centroid, int *device_nearest_centroid, int *device_change, int *count) {
    extern __shared__ float shared_centroids[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int shared_len = k * d;
    for (int i = threadIdx.x; i < shared_len; i += blockDim.x) {
        shared_centroids[i] = device_centroids[i];
    }
    __syncthreads();
    if (idx >= N) return;
    // use shared memory
    // if(idx==2){
    //     printf("Shared Centroids: \n");
    //     for(int i=0;i<k;i++){
    //         for(int j=0;j<d;j++){
    //             printf("%f ", shared_centroids(i, j));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    int x_local[100];
    for (int j = 0; j < d; j++) {
        x_local[j] = x(idx, j);  // Coalesced global read
        //printf("pt: %d, dim: %d, val: %lf\n", idx, j, x_local[j]);
    }
    double mn_dist = DBL_MAX;
    int mn_dist_cent = -1;
    for (int i = 0; i < k; i++) {
        //printf("i:%d, (2,0):%lf\n",i, shared_centroids(2,0));
        double curr_dist = 0;
        for (int j = 0; j < d; j++) {
            double diff = (x_local[j] - shared_centroids(i, j));
            curr_dist += diff * diff;
            // if(i==0 && idx==2){
            //     printf("j %d, diff: %lf, x[j] %lf, cent[j] %lf\n", j, diff, x_local[j], shared_centroids(i, j));
            // }
        }
        if (curr_dist < mn_dist) {
            //printf("Distance: %f for cent: %d, pt: %d\n", curr_dist, i, idx);
            mn_dist = curr_dist;
            mn_dist_cent = i;
        }
    }
    device_new_nearest_centroid[idx] = mn_dist_cent;
    int old_centroid = device_nearest_centroid[idx];
    if (old_centroid != mn_dist_cent) {
        //printf("Change: %d to %d for %d\n", old_centroid, mn_dist_cent, idx);
        *device_change = 1;
        atomicAdd(&count[mn_dist_cent], 1);
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

    int ii = (i-1) >= 0 ? count_prefix[i-1] : 0;

    for (int j=0; j<d; j++){
        device_centroids_acc((ii+off), j) = x(idx, j);
        // printf("device_centroids_acc[%d][%d] = [%d] = %f\n", ii+off, j, (ii+off)*d + j,device_centroids_acc((ii+off), j));
        // printf("x[%d][%d] = %f\n", idx, j, x(idx, j));
    }
}

// cuML implementation of k-means cross check

// launch on threads = number of centroids
__global__ void update_k_means_2(double *device_centroids_acc, double *device_centroids, int N, int d, int k, int *count_prefix, int *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    int end_idx = count_prefix[idx]-1;
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
            if(i<k) {
                centroids[i+j*k] = x(i, j);
            }
        }
    }

    // start time
    auto start_total = std::chrono::high_resolution_clock::now();

    int *device_x;
    double *device_centroids;
    int *device_nearest_centroid, *device_new_nearest_centroid;
    int *y = (int *)malloc(N * sizeof(int));
    int *device_change, change;
    int *device_count, *device_count_prefix, *count_prefix = (int *)malloc(k * sizeof(int));
    int *temp_int;
    double *temp_double;
    double *device_centroids_acc;
    int *count = (int *)malloc(k * sizeof(int));

    cudaMalloc(&temp_int, k * sizeof(int));
    cudaMalloc(&temp_double, N * d * sizeof(double));

    cudaMalloc(&device_x, d * N * sizeof(int));
    cudaMalloc(&device_centroids, d * k * sizeof(double));
    cudaMalloc(&device_nearest_centroid, N * sizeof(int));
    cudaMalloc(&device_new_nearest_centroid, N * sizeof(int));
    cudaMalloc(&device_change, sizeof(int));
    cudaMalloc(&device_count, k * sizeof(int));
    cudaMalloc(&device_count_prefix, k * sizeof(int));
    cudaMalloc(&device_centroids_acc, N*d*sizeof(double));

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
    assign_centroid<<<NUM_BLOCKS_PTS, NUM_THREADS_PTS, k * d * sizeof(float)>>>(device_x, N, d, k, device_centroids, device_new_nearest_centroid, device_nearest_centroid, device_change, device_count);
    cudaMemcpy(&change, device_change, sizeof(int), cudaMemcpyDeviceToHost);

    // cudaMemcpy(centroids, device_centroids, k * d * sizeof(double), cudaMemcpyDeviceToHost);
    // cout << "Centroids: ";
    // for(int i=0;i<k;i++){
    //     for(int j=0;j<d;j++){
    //         cout << centroids[i+j*k] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;
    while (change) {
        cudaMemcpy(device_nearest_centroid, device_new_nearest_centroid, N * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(device_centroids, 0, d * k * sizeof(double));
        cudaMemcpy(device_count_prefix, device_count, k * sizeof(int), cudaMemcpyDeviceToDevice);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed 1: %s\n", cudaGetErrorString(err));
        }

        PREFIX_SUM_INT(device_count_prefix, temp_int, k);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed 2: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(count_prefix, device_count_prefix, k * sizeof(int), cudaMemcpyDeviceToHost);
        
        
        // cout << "Count New: ";
        // for(int i=0;i<k;i++){
        //     cout << count_prefix[i] << " ";
        // }
        // cout << endl;
        // cudaMemcpy(y, device_nearest_centroid, N * sizeof(int), cudaMemcpyDeviceToHost);
        // cout << "Nearest Centroid: \n";
        // for(int i=0;i<N;i++){
        //     cout << y[i] << " ";
        // }
        // cout << endl;

        cudaMemset(temp_int, 0, k * sizeof(int));

        update_k_means_1<<<NUM_BLOCKS_PTS, NUM_THREADS_PTS>>>(device_centroids_acc, device_x, N, d, k, device_nearest_centroid, device_count_prefix, temp_int);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed 3: %s\n", cudaGetErrorString(err));
        }
        
        // double centroid_acc[N*d];
        // cudaMemcpy(centroid_acc, device_centroids_acc, N * d * sizeof(double), cudaMemcpyDeviceToHost);
        // cout << "Centroid Acc before: \n";
        // for(int i=0;i<N;i++){
        //     for(int j=0;j<d;j++){
        //         cout << centroid_acc[i*d+j] << " ";
        //     }
        //     cout << endl;
        // }
        // cout << endl;

        for(int i=0 ; i<k ; i++){
            // cout << "i: " << i << endl;
            int curr = ((i-1) >= 0 ? count_prefix[i-1] : 0);
            int n = (count_prefix[i]-curr);
            PREFIX_SUM_DOUBLE((device_centroids_acc + curr*d), temp_double, n, d);
            // err = cudaGetLastError();
            // if (err != cudaSuccess) {
            //     printf("Kernel launch failed 4: %s\n", cudaGetErrorString(err));
            // }
        }

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

        assign_centroid<<<NUM_BLOCKS_PTS, NUM_THREADS_PTS, k * d * sizeof(float)>>>(device_x, N, d, k, device_centroids, device_new_nearest_centroid, device_nearest_centroid, device_change, device_count);
        // err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     printf("Kernel launch failed 6: %s\n", cudaGetErrorString(err));
        // }

        
        // cudaMemcpy(centroids, device_centroids, k * d * sizeof(double), cudaMemcpyDeviceToHost);
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
    out << "Total: " << elapsed_total.count() * 1000 << "ms, Calculation Time: " << elapsed_calc.count()*1000 << "ms";
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