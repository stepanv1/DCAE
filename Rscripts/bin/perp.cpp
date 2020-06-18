#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <omp.h>
#include <limits>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <float.h>
#include <stdlib.h>
//#include "vptree.h"
//#include "sptree.h"
//#include "tsne.h"

#define NUM_THREADS(Nt) ((Nt) >= 0 ? (Nt) : omp_get_num_procs() + (Nt) + 1)
#define DBL_MAX __DBL_MAX__
#define DBL_MIN __DBL_MIN__
//tsne_out <- Rtsne(matrix(rnorm(10000*10),10000,10), num_threads=12) # Run TSNE

using namespace std;

// compile:
//  g++ -fPIC -c -Wall perp.cpp
// ld -shared perp.o -o perp.so
// this will the the shared library perp.so to plug in python code for DA
// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
// K number of mearest neighbors
// N number of data points
// D dimensions 
// _row_P
// _col_P
// _val_P
// perplexity - perplexity
// void TSNE<treeT, dist_fn>::computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, int K, int verbose) {

extern "C"{ 
void Perplexity( double* dist,  int N, int D,   double* P, double perplexity,  int K, double *Sigma, int num_threads) {

    if(perplexity > K) printf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    unsigned int* row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
    //double* val_P = (double*) calloc(N * K, sizeof(double));
    double* cur_P = (double*) malloc(K * sizeof(double));
    if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int) K;

   
    printf("Computing sigma...\n");
//#pragma omp parallel for
    for(int n = 0; n < N; n++) {

        if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);

        // Find nearest neighbors
        //int *indices = (ind+n);
        double *distances = (dist);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0; double sum_P;
        while(!found && iter < 200) {

            // Compute Gaussian kernel row
            for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m] * distances[m]);

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for(int m = 0; m < K; m++) sum_P += cur_P[m];
            double H = .0;
            for(int m = 0; m < K; m++) H += beta * (distances[m] * distances[m] * cur_P[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if(Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if(Hdiff > 0) {
                    min_beta = beta;
                    if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for(unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(unsigned int m = 0; m < K; m++) {
            //col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
            P[row_P[n] + m] = cur_P[m];
        }

        Sigma[n] = 0.5/beta;
    }

    // Clean up memory
    //obj_X.clear();
    free(cur_P);
}
}