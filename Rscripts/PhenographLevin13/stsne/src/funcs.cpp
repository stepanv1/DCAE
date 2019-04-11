//#include "vptree.h"
#include "vptree.h"
#include <Rcpp.h>


using namespace Rcpp;

/*RcppExport SEXP tree__new() {
  // convert inputs to appropriate C++ types
  // create a pointer to an Uniform object and wrap it
  // as an external pointer
  Rcpp::XPtr<VpTree<DataPoint, euclidean_distance>> ptr(new VpTree<DataPoint, euclidean_distance>::VpTree(), true );
  // return the external pointer to the R side
  return ptr;
}*/
//data=matrix(rnorm(10000), nrow=10)
// [[Rcpp::export]]
RcppExport SEXP tree_create(SEXP data_,  SEXP num_threads_){  
  int num_threads = as<int>(num_threads_);
  
  
  std::cout << "Generating euclidean tree.... 1 " << std::endl;
  NumericMatrix d(data_);
  
  int origN = d.nrow();
  int D = d.ncol();
  double  *dat_;
  std::cout << "Generating euclidean tree.... 2 " << std::endl;
  dat_ = (double*) calloc(D * origN, sizeof(double));
  if(dat_ == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
  for (unsigned int i = 0; i < origN; i++){
    for (unsigned int j = 0; j < D; j++){
      dat_[i*D+j] = d(i,j);
    }
  }
  
  VpTree<DataPoint, euclidean_distance> *tree = new VpTree<DataPoint, euclidean_distance>();
  std::cout << "Generating euclidean tree.... 3 " << std::endl;
  VPSearchFunc::createEuclideanTree(&dat_, &tree,  origN, D, num_threads);
  //printf("Address of tree1 is %p\n", (void *)tree);  
  //free(data_); data_ = NULL;
  free(dat_); dat_ = NULL;
  
  return Rcpp::XPtr<VpTree<DataPoint, euclidean_distance>> (tree, true);
}

// [[Rcpp::export]]
Rcpp::List tree_search(SEXP tree_,  SEXP data_, SEXP num_neighbors_,
                             SEXP num_threads_){  
  //std::cout<< tree_ <<std::endl; 
  int num_threads = as<int>(num_threads_);
  int num_neighbors = as<int>(num_neighbors_);
  NumericMatrix d(data_);
  //Rcpp::XPtr<VpTree<DataPoint, euclidean_distance>> tr(tree_);
  VpTree<DataPoint, euclidean_distance>* tr = Rcpp::as<Rcpp::XPtr<VpTree<DataPoint, euclidean_distance>>>(tree_);
  //printf("Address of tree2 is %p\n", tr);  
  
  int origN = d.nrow();
  int D = d.ncol();
  double  *dat_;
  std::cout << "Generating euclidean tree.... 2 " << std::endl;
  dat_ = (double*) calloc(D * origN, sizeof(double));
  if(dat_ == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
  for (unsigned int i = 0; i < origN; i++){
    for (unsigned int j = 0; j < D; j++){
      dat_[i*D+j] = d(i,j);
    }
  }
  
  int * indices_t2_matrix;
  double * distances_t2_matrix;
  distances_t2_matrix = (double*)calloc(origN * num_neighbors, sizeof(double));
  indices_t2_matrix = (int*)calloc(origN * num_neighbors, sizeof(int));
  
  std::cout << "Searching .... 3 " << std::endl;
  
  /*do 
  {
    std::cout << '\n' << "Press a key to continue...";
  } while (std::cin.get() != '\n');
  */
  VPSearchFunc::findNearestNeighborsMatrix(&dat_, &tr, &distances_t2_matrix,
                                           &indices_t2_matrix, D, origN, num_neighbors, num_threads);
  /*
  do 
  {
    std::cout << '\n' << "Press a key to continue...";
  } while (std::cin.get() != '\n');
  */
  
  Rcpp::NumericMatrix dist(origN, num_neighbors);
  for (int i = 0; i < origN; i++){
    for (int j = 0; j < num_neighbors; j++){
      dist(i,j) = distances_t2_matrix[i*num_neighbors+j];
    }
  }
  Rcpp::NumericMatrix idx(origN, num_neighbors);
  for (int i = 0; i < origN; i++){
    for (int j = 0; j < num_neighbors; j++){
      idx(i,j) = indices_t2_matrix[i*num_neighbors+j]+1;
    }
  }
  
  Rcpp::List output = Rcpp::List::create(Rcpp::_["dist"]= dist,
                                         Rcpp::_["idx"]= idx);
  
  //free(data_); data_ = NULL;
  free(dat_); dat_ = NULL;
  free(distances_t2_matrix); distances_t2_matrix = NULL;
  free(indices_t2_matrix); indices_t2_matrix = NULL;
  
  /*
  do 
   {
   std::cout << '\n' << "Press a key to continue...";
   } while (std::cin.get() != '\n');
  */
  return output;
}

// [[Rcpp::export]]
Rcpp::List matrix_search(SEXP data_, SEXP num_neighbors_,
                            SEXP num_threads_){  
  std::cout << "Start\n---------------------------10" << std::endl;
  int num_threads = as<int>(num_threads_);
  int num_neighbors = as<int>(num_neighbors_);
  NumericMatrix d(data_);
  
  std::cout << "Start\n---------------------------11" << std::endl;
  int origN = d.nrow();
  int D = d.ncol();
  std::cout << origN << std::endl;
  std::cout << D << std::endl;
  double  *dat_;
  std::cout << "Generating euclidean tree.... 2 " << std::endl;
  dat_ = (double*) calloc(D * origN, sizeof(double));
  if(dat_ == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
  for (unsigned int i = 0; i < origN; i++){
    for (unsigned int j = 0; j < D; j++){
      dat_[i*D+j] = d(i,j);
    }
  }
  std::cout << "Start\n---------------------------12" << std::endl;
  int * indices_t2_matrix;
  double * distances_t2_matrix;
  distances_t2_matrix = (double*)calloc(origN * num_neighbors, sizeof(double));
  //double* distance_matrix = *distances;
  
  indices_t2_matrix = (int*)calloc(origN * num_neighbors, sizeof(int));
  //int* indices_matrix = *indices;
  
  
  //VPSearchFunc::printMatrix(&dat_, origN, D);
  
  /*do 
  {
    std::cout << '\n' << "Press a key to continue...";
  } while (std::cin.get() != '\n');
  */
  
  std::cout << "Searching .... 3 " << std::endl;
  VPSearchFunc::findNearestNeighborsTarget(&dat_, &dat_, &distances_t2_matrix,
                                           &indices_t2_matrix, D, origN, origN, num_neighbors, num_threads);
  
  //VPSearchFunc::printMatrix(&distances_t2_matrix, origN, num_neighbors);
  //VPSearchFunc::printMatrixInt(&indices_t2_matrix, origN, num_neighbors);
  /*do 
  {
    std::cout << '\n' << "Press a key to continue...";
  } while (std::cin.get() != '\n');
  */
  Rcpp::NumericMatrix dist(origN, num_neighbors);
  for (int i = 0; i < origN; i++){
    for (int j = 0; j < num_neighbors; j++){
      dist(i,j) = distances_t2_matrix[i*num_neighbors+j];
    }
  }
  Rcpp::NumericMatrix idx(origN, num_neighbors);
  for (int i = 0; i < origN; i++){
    for (int j = 0; j < num_neighbors; j++){
      idx(i,j) = indices_t2_matrix[i*num_neighbors+j]+1;// add 1 index for R compatibility 
    }
  }
  
  Rcpp::List output = Rcpp::List::create(Rcpp::_["dist"]= dist,
                                         Rcpp::_["idx"]= idx);
  
  //free(data_); data_ = NULL;
  free(dat_); dat_ = NULL;
  free(distances_t2_matrix); distances_t2_matrix = NULL;
  free(indices_t2_matrix); indices_t2_matrix = NULL;
 
  
  /*do 
   {
   std::cout << '\n' << "Press a key to continue...";
   } while (std::cin.get() != '\n');
  */
  return output;
}





//data=matrix(rnorm(10000), ncol=10)
//data=matrix(rnorm(10000), ncol=10)
//tr<-tree_create(data,  1)
//tree_search(tr,  data, 3, 1)
//t1<-matrix_search(data, 3, 1)
//t2<-matrix_search(data, 3, 1)  
//View(t1$idx);View(t2$idx)
  
//tr1<-tree_create(data,  1)
//tm1<-tree_search(tr1, data, 3, 1) 
//tr2<-tree_create(data,  1)
//tm2<-tree_search(tr2, data, 3, 1)   
//View(tm1$idx);View(tm2$idx)    
  
//system.time(z<-matrix_search(data[[3]],system.time(z<-tree_search(tr, data[[3]], 30, 12)) 30, 12))
//tr<-tree_create(data[[3]],  10)
//system.time(z<-tree_search(tr, data[[3]], 30, 12))
//system.time(z<-tree_search(tr, data[1:10000,], 30, 12))
//system.time(z<-Rtsne( data[1:10000,], num_threads=10))
