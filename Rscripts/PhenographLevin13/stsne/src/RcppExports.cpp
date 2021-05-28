// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// tree_create
RcppExport SEXP tree_create(SEXP data_, SEXP num_threads_);
RcppExport SEXP _stsne_tree_create(SEXP data_SEXP, SEXP num_threads_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type data_(data_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type num_threads_(num_threads_SEXP);
    rcpp_result_gen = Rcpp::wrap(tree_create(data_, num_threads_));
    return rcpp_result_gen;
END_RCPP
}
// tree_search
Rcpp::List tree_search(SEXP tree_, SEXP data_, SEXP num_neighbors_, SEXP num_threads_);
RcppExport SEXP _stsne_tree_search(SEXP tree_SEXP, SEXP data_SEXP, SEXP num_neighbors_SEXP, SEXP num_threads_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type tree_(tree_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type data_(data_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type num_neighbors_(num_neighbors_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type num_threads_(num_threads_SEXP);
    rcpp_result_gen = Rcpp::wrap(tree_search(tree_, data_, num_neighbors_, num_threads_));
    return rcpp_result_gen;
END_RCPP
}
// matrix_search
Rcpp::List matrix_search(SEXP data_, SEXP num_neighbors_, SEXP num_threads_);
RcppExport SEXP _stsne_matrix_search(SEXP data_SEXP, SEXP num_neighbors_SEXP, SEXP num_threads_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type data_(data_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type num_neighbors_(num_neighbors_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type num_threads_(num_threads_SEXP);
    rcpp_result_gen = Rcpp::wrap(matrix_search(data_, num_neighbors_, num_threads_));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP _rcpp_module_boot_NumEx();

static const R_CallMethodDef CallEntries[] = {
    {"_stsne_tree_create", (DL_FUNC) &_stsne_tree_create, 2},
    {"_stsne_tree_search", (DL_FUNC) &_stsne_tree_search, 4},
    {"_stsne_matrix_search", (DL_FUNC) &_stsne_matrix_search, 3},
    {"_rcpp_module_boot_NumEx", (DL_FUNC) &_rcpp_module_boot_NumEx, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_stsne(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
