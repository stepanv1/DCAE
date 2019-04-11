#include "vptree.h"

extern "C"
{
  extern void findNearestNeighbors(double* data, double* distances,
                                              int * indices, int dim, int num_rows,
                                              int num_neighbors, int num_threads){
    VPSearchFunc::findNearestNeighborsPy(&data, &distances,&indices, dim, num_rows, num_neighbors, num_threads);
  }

  extern void findNearestNeighborsTarget(double* data, double* data_target, double* distances,
                                              int * indices, int dim, int num_rows,
                                              int num_rows_target,
                                              int num_neighbors, int num_threads){
    VPSearchFunc::findNearestNeighborsTargetPy(&data, &data_target, &distances,&indices, dim, num_rows, num_rows_target, num_neighbors, num_threads);
  }
}
