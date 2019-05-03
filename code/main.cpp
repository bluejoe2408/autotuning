#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <algorithm>
#include "mergeSort_common.h"


////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // host variable
    uint *h_SrcKey, *h_SrcVal, *h_DstKey, *h_DstVal;
    uint *hm_SrcKey, *hm_SrcVal, *hm_BufKey, *hm_BufVal, *hm_DstKey, *hm_DstVal;
    uint *hq_SrcKey, *hq_SrcVal, *hq_BufKey, *hq_BufVal, *hq_DstKey, *hq_DstVal;
    // device variable
    uint *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;
    uint *db_SrcKey, *db_SrcVal, *db_BufKey, *db_BufVal, *db_DstKey, *db_DstVal;
    // test var
    uint keysFlag, valuesFlag;

    StopWatchInterface *hTimer = NULL;

    uint N = 1024;
    const uint DIR = 1;
    const uint numValues = 32767;

    printf("%s Starting...\n\n", argv[0]);

    int dev = findCudaDevice(argc, (const char **) argv);

    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

    /* pending */
    int tmpN = 1;
    while(tmpN < N) tmpN <<= 1;

    printf("Allocating and initializing host arrays...\n\n");
    sdkCreateTimer(&hTimer);
    h_SrcKey = (uint *)malloc(tmpN * sizeof(uint));
    h_SrcVal = (uint *)malloc(tmpN * sizeof(uint));
    h_DstKey = (uint *)malloc(tmpN * sizeof(uint));
    h_DstVal = (uint *)malloc(tmpN * sizeof(uint));

    srand(2019);

    for (uint i = 0; i < N; i++)
    {
        h_SrcKey[i] = rand() % numValues;
    }


    /* pending add 0*/
    for(uint i = N; i < tmpN; i++ ) h_DstKey[i] = 0;
    N = tmpN;

    fillValues(h_SrcVal, N);

    printf("Allocating and initializing host merge sort arrays...\n\n");
    hm_SrcKey = (uint *)malloc(N * sizeof(uint));
    hm_SrcVal = (uint *)malloc(N * sizeof(uint));
    hm_BufKey = (uint *)malloc(N * sizeof(uint));
    hm_BufVal = (uint *)malloc(N * sizeof(uint));
    hm_DstKey = (uint *)malloc(N * sizeof(uint));
    hm_DstVal = (uint *)malloc(N * sizeof(uint));
    memcpy(hm_SrcKey, h_SrcKey, N * sizeof(uint));
    memcpy(hm_SrcVal, h_SrcVal, N * sizeof(uint));


    printf("Running CPU merge sort...\n");
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    mergeSortHost(
        hm_DstKey,
        hm_DstVal,
        hm_BufKey,
        hm_BufVal,
        hm_SrcKey,
        hm_SrcVal,
        N,
        DIR
    );
    sdkStopTimer(&hTimer);
    printf("Time: %f ms\n", sdkGetTimerValue(&hTimer));
    printf("Inspecting the results...\n");
    keysFlag = validateSortedKeys(
                        hm_DstKey,
                        hm_SrcKey,
                        1,
                        N,
                        numValues,
                        DIR
                    );

    valuesFlag = validateSortedValues(
                          hm_DstKey,
                          hm_DstVal,
                          hm_SrcKey,
                          1,
                          N
                      );


    printf("\nAllocating and initializing host quick sort arrays...\n\n");
    hq_SrcKey = (uint *)malloc(N * sizeof(uint));
    hq_SrcVal = (uint *)malloc(N * sizeof(uint));
    hq_BufKey = (uint *)malloc(N * sizeof(uint));
    hq_BufVal = (uint *)malloc(N * sizeof(uint));
    hq_DstKey = (uint *)malloc(N * sizeof(uint));
    hq_DstVal = (uint *)malloc(N * sizeof(uint));
    memcpy(hq_SrcKey, h_SrcKey, N * sizeof(uint));
    memcpy(hq_SrcVal, h_SrcVal, N * sizeof(uint));

    printf("Running CPU quick sort...\n");
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    std::sort(hq_SrcKey, hq_SrcKey + N);
    sdkStopTimer(&hTimer);
    printf("Time: %f ms\n", sdkGetTimerValue(&hTimer));

    printf("\nAllocating and initializing CUDA merge arrays...\n\n");
    checkCudaErrors(cudaMalloc((void **)&d_DstKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_DstVal, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_BufKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_BufVal, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_SrcKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_SrcVal, N * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint), cudaMemcpyHostToDevice));
    printf("Allocating and initializing CUDA bitonic arrays...\n\n");
    checkCudaErrors(cudaMalloc((void **)&db_DstKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&db_DstVal, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&db_BufKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&db_BufVal, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&db_SrcKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&db_SrcVal, N * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(db_SrcKey, h_SrcKey, N * sizeof(uint), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(db_SrcVal, h_SrcVal, N * sizeof(uint), cudaMemcpyHostToDevice));


    printf("Initializing GPU merge sort...\n");
    initMergeSort();

    printf("Running GPU merge sort...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    mergeSort(
        d_DstKey,
        d_DstVal,
        d_BufKey,
        d_BufVal,
        d_SrcKey,
        d_SrcVal,
        N,
        DIR
    );
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    printf("Time: %f ms\n", sdkGetTimerValue(&hTimer));

    printf("Reading back GPU merge sort results...\n");
    checkCudaErrors(cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint), cudaMemcpyDeviceToHost));

    printf("Inspecting the results...\n");
    keysFlag = validateSortedKeys(
                        h_DstKey,
                        h_SrcKey,
                        1,
                        N,
                        numValues,
                        DIR
                    );

    valuesFlag = validateSortedValues(
                          h_DstKey,
                          h_DstVal,
                          h_SrcKey,
                          1,
                          N
                      );

    printf("Shutting down...\n");
    closeMergeSort();

    printf("Initializing GPU bitonic sort...\n");
    initMergeSort();

    printf("Running GPU bitonic sort...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    bitonicSort(
        db_DstKey,
        db_DstVal,
        db_BufKey,
        db_BufVal,
        db_SrcKey,
        db_SrcVal,
        N,
        DIR
    );
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    printf("Time: %f ms\n", sdkGetTimerValue(&hTimer));

    printf("Reading back GPU bitonic sort results...\n");
    checkCudaErrors(cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint), cudaMemcpyDeviceToHost));

    printf("Inspecting the results...\n");
    keysFlag = validateSortedKeys(
                        h_DstKey,
                        h_SrcKey,
                        1,
                        N,
                        numValues,
                        DIR
                    );

    valuesFlag = validateSortedValues(
                          h_DstKey,
                          h_DstVal,
                          h_SrcKey,
                          1,
                          N
                      );

    printf("Shutting down...\n");
    closeMergeSort();

    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cudaFree(d_SrcVal));
    checkCudaErrors(cudaFree(d_SrcKey));
    checkCudaErrors(cudaFree(d_BufVal));
    checkCudaErrors(cudaFree(d_BufKey));
    checkCudaErrors(cudaFree(d_DstVal));
    checkCudaErrors(cudaFree(d_DstKey));
    free(h_DstVal);
    free(h_DstKey);
    free(h_SrcVal);
    free(h_SrcKey);

    exit((keysFlag && valuesFlag) ? EXIT_SUCCESS : EXIT_FAILURE);
}
