#define ROUND 21501
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
    uint *hb_SrcKey, *hb_SrcVal, *hb_DstKey, *hb_DstVal;
    uint *hm_SrcKey, *hm_SrcVal, *hm_BufKey, *hm_BufVal, *hm_DstKey, *hm_DstVal;
    uint *hq_SrcKey, *hq_SrcVal, *hq_BufKey, *hq_BufVal, *hq_DstKey, *hq_DstVal;
    // device variable
    uint *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;
    uint *db_SrcKey, *db_SrcVal, *db_BufKey, *db_BufVal, *db_DstKey, *db_DstVal;
    // test var
    uint keysFlag, valuesFlag;
    // remove the first gpu call
    uint R1;
    FILE *fp = NULL;
    FILE *fpp = NULL;
    FILE *fppp = NULL;
    fp = fopen("average.m", "w");
    fpp = fopen("variance.m","w");
    fppp = fopen("standard_deviation.m","w");
    fprintf(fp,"x=1:100:%d;",ROUND);
    fprintf(fp,"y1=1:100:%d;",ROUND);
    fprintf(fp,"y2=1:100:%d;",ROUND);
    fprintf(fp,"y3=1:100:%d;",ROUND);
    fprintf(fp,"y4=1:100:%d;\n",ROUND);
    fprintf(fpp,"fid = fopen('variance.txt','wt');\n");
    fprintf(fppp,"fid = fopen('standard_deviation.txt','wt');\n");
    //printf("%s Starting...\n\n", argv[0]);

    int dev = findCudaDevice(argc, (const char **) argv);

    if (dev == -1) {
        return EXIT_FAILURE;
    }

    for(uint NUM = 1; NUM <= ROUND; NUM+=100) {
        fprintf(fpp,"x1 = 1:10;");
        fprintf(fpp,"x2 = 1:10;");
        fprintf(fpp,"x3 = 1:9;");
        fprintf(fpp,"x4 = 1:10;\n");
        fprintf(fppp,"x1 = 1:10;");
        fprintf(fppp,"x2 = 1:10;");
        fprintf(fppp,"x3 = 1:9;");
        fprintf(fppp,"x4 = 1:10;\n");
        StopWatchInterface *hTimer = NULL;
        float t1 = 0, t2 = 0, t3 = 0, t4 = 0;

        uint N = NUM;
        const uint DIR = 1;
        const uint numValues = 32767;



        /* pending */
        int tmpN = 1;
        while (tmpN < N) tmpN <<= 1;
        if (tmpN < 1024) tmpN = 1024;

        //printf("Allocating and initializing host arrays...\n\n");
        sdkCreateTimer(&hTimer);
        h_SrcKey = (uint *) malloc(tmpN * sizeof(uint));
        h_SrcVal = (uint *) malloc(tmpN * sizeof(uint));
        h_DstKey = (uint *) malloc(tmpN * sizeof(uint));
        h_DstVal = (uint *) malloc(tmpN * sizeof(uint));
        hb_SrcKey = (uint *) malloc(tmpN * sizeof(uint));
        hb_SrcVal = (uint *) malloc(tmpN * sizeof(uint));
        hb_DstKey = (uint *) malloc(tmpN * sizeof(uint));
        hb_DstVal = (uint *) malloc(tmpN * sizeof(uint));

        srand(2019);

        for (uint loop = 0; loop < 10; loop++) {

            for (uint i = 0; i < N; i++) {
                h_SrcKey[i] = rand() % numValues;
            }

            /* pending add 0*/
            for (uint i = N; i < tmpN; i++) h_SrcKey[i] = 0;
            N = tmpN;

            fillValues(h_SrcVal, N);

            for (uint i = 0; i < N; i++) {
                hb_SrcKey[i] = h_SrcKey[i];
                hb_SrcVal[i] = h_SrcVal[i];
            }


            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
            //printf("Allocating and initializing host merge sort arrays...\n\n");
            hm_SrcKey = (uint *) malloc(NUM * sizeof(uint));
            hm_SrcVal = (uint *) malloc(NUM * sizeof(uint));
            hm_BufKey = (uint *) malloc(NUM * sizeof(uint));
            hm_BufVal = (uint *) malloc(NUM * sizeof(uint));
            hm_DstKey = (uint *) malloc(NUM * sizeof(uint));
            hm_DstVal = (uint *) malloc(NUM * sizeof(uint));
            memcpy(hm_SrcKey, h_SrcKey, NUM * sizeof(uint));
            memcpy(hm_SrcVal, h_SrcVal, NUM * sizeof(uint));
            //printf("Running CPU merge sort...\n");
            mergeSortHost(
                    hm_DstKey,
                    hm_DstVal,
                    hm_BufKey,
                    hm_BufVal,
                    hm_SrcKey,
                    hm_SrcVal,
                    NUM,
                    DIR
            );
            sdkStopTimer(&hTimer);
            t1 += sdkGetTimerValue(&hTimer);
            //printf("Inspecting the results...\n");
            fprintf(fpp,"x1(%d) = %f; ",loop+1, sdkGetTimerValue(&hTimer));
            fprintf(fppp,"x1(%d) = %f; ",loop+1, sdkGetTimerValue(&hTimer));
            keysFlag = validateSortedKeys(
                    hm_DstKey,
                    hm_SrcKey,
                    1,
                    NUM,
                    numValues,
                    DIR
            );

            valuesFlag = validateSortedValues(
                    hm_DstKey,
                    hm_DstVal,
                    hm_SrcKey,
                    1,
                    NUM
            );


            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
            //printf("\nAllocating and initializing host quick sort arrays...\n\n");
            hq_SrcKey = (uint *) malloc(NUM * sizeof(uint));
            hq_SrcVal = (uint *) malloc(NUM * sizeof(uint));
            hq_BufKey = (uint *) malloc(NUM * sizeof(uint));
            hq_BufVal = (uint *) malloc(NUM * sizeof(uint));
            hq_DstKey = (uint *) malloc(NUM * sizeof(uint));
            hq_DstVal = (uint *) malloc(NUM * sizeof(uint));
            memcpy(hq_SrcKey, h_SrcKey, NUM * sizeof(uint));
            memcpy(hq_SrcVal, h_SrcVal, NUM * sizeof(uint));

            //printf("Running CPU quick sort...\n");
            std::sort(hq_SrcKey, hq_SrcKey + NUM);
            sdkStopTimer(&hTimer);
            t2 += sdkGetTimerValue(&hTimer);
            fprintf(fpp,"x2(%d) = %f; ",loop+1, sdkGetTimerValue(&hTimer));
            fprintf(fppp,"x2(%d) = %f; ",loop+1, sdkGetTimerValue(&hTimer));

            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
            //printf("\nAllocating and initializing CUDA merge arrays...\n\n");
            checkCudaErrors(cudaMalloc((void **) &d_DstKey, N * sizeof(uint)));
            checkCudaErrors(cudaMalloc((void **) &d_DstVal, N * sizeof(uint)));
            checkCudaErrors(cudaMalloc((void **) &d_BufKey, N * sizeof(uint)));
            checkCudaErrors(cudaMalloc((void **) &d_BufVal, N * sizeof(uint)));
            checkCudaErrors(cudaMalloc((void **) &d_SrcKey, N * sizeof(uint)));
            checkCudaErrors(cudaMalloc((void **) &d_SrcVal, N * sizeof(uint)));
            checkCudaErrors(cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint), cudaMemcpyHostToDevice));
            //printf("Initializing GPU merge sort...\n");
            initMergeSort();
            //printf("Running GPU merge sort...\n");
            checkCudaErrors(cudaDeviceSynchronize());
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
            //printf("Reading back GPU merge sort results...\n");
            checkCudaErrors(cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint), cudaMemcpyDeviceToHost));
            sdkStopTimer(&hTimer);
            t3 += sdkGetTimerValue(&hTimer);
            if(loop>0){
            fprintf(fpp,"x3(%d) = %f; ",loop, sdkGetTimerValue(&hTimer));
            fprintf(fppp,"x3(%d) = %f; ",loop, sdkGetTimerValue(&hTimer));}
            if (loop == 0) R1 = sdkGetTimerValue(&hTimer);
            //printf("Inspecting the results...\n");
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
            //printf("Shutting down...\n");
            closeMergeSort();

            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
            //printf("Allocating and initializing CUDA bitonic arrays...\n\n");
            checkCudaErrors(cudaMalloc((void **) &db_DstKey, N * sizeof(uint)));
            checkCudaErrors(cudaMalloc((void **) &db_DstVal, N * sizeof(uint)));
            checkCudaErrors(cudaMalloc((void **) &db_BufKey, N * sizeof(uint)));
            checkCudaErrors(cudaMalloc((void **) &db_BufVal, N * sizeof(uint)));
            checkCudaErrors(cudaMalloc((void **) &db_SrcKey, N * sizeof(uint)));
            checkCudaErrors(cudaMalloc((void **) &db_SrcVal, N * sizeof(uint)));
            checkCudaErrors(cudaMemcpy(db_SrcKey, hb_SrcKey, N * sizeof(uint), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(db_SrcVal, hb_SrcVal, N * sizeof(uint), cudaMemcpyHostToDevice));
            //printf("\nInitializing GPU bitonic sort...\n");
            initMergeSort();
            //printf("Running GPU bitonic sort...\n");
            checkCudaErrors(cudaDeviceSynchronize());
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
            //printf("Reading back GPU bitonic sort results...\n");
            checkCudaErrors(cudaMemcpy(h_DstKey, db_DstKey, N * sizeof(uint), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_DstVal, db_DstVal, N * sizeof(uint), cudaMemcpyDeviceToHost));
            sdkStopTimer(&hTimer);
            t4 += sdkGetTimerValue(&hTimer);
            fprintf(fpp,"x4(%d) = %f; \n",loop+1, sdkGetTimerValue(&hTimer));
            fprintf(fppp,"x4(%d) = %f; \n",loop+1, sdkGetTimerValue(&hTimer));
            //printf("Inspecting the results...\n");
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
            //printf("Shutting down...\n");
            closeMergeSort();

            fprintf(fpp,"fprintf(fid,'1 %%f %%f %%f %%f\\n',var(x1),var(x2),var(x3),var(x4));\n");
            fprintf(fppp,"fprintf(fid,'1 %%f %%f %%f %%f\\n',std(x1),std(x2),std(x3),std(x4));\n");

        }
        fprintf(fp,"y1(%d) = %f; ",1+NUM/100, t1/10);
        fprintf(fp,"y2(%d) = %f; ",1+NUM/100, t2/10);
        fprintf(fp,"y3(%d) = %f; ",1+NUM/100, (t3 - R1) / 9);
        fprintf(fp,"y4(%d) = %f;\n",1+NUM/100, t4 / 10);

        //finally release the space
        sdkDeleteTimer(&hTimer);
        checkCudaErrors(cudaFree(d_SrcVal));
        checkCudaErrors(cudaFree(d_SrcKey));
        checkCudaErrors(cudaFree(d_BufVal));
        checkCudaErrors(cudaFree(d_BufKey));
        checkCudaErrors(cudaFree(d_DstVal));
        checkCudaErrors(cudaFree(d_DstKey));
        checkCudaErrors(cudaFree(db_SrcVal));
        checkCudaErrors(cudaFree(db_SrcKey));
        checkCudaErrors(cudaFree(db_BufVal));
        checkCudaErrors(cudaFree(db_BufKey));
        checkCudaErrors(cudaFree(db_DstVal));
        checkCudaErrors(cudaFree(db_DstKey));
        free(h_DstVal);
        free(h_DstKey);
        free(h_SrcVal);
        free(h_SrcKey);
        free(hb_DstVal);
        free(hb_DstKey);
        free(hb_SrcVal);
        free(hb_SrcKey);

        //exit((keysFlag && valuesFlag) ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    fprintf(fp,"plot(x,y1,x,y2,x,y3,x,y4);");
}
