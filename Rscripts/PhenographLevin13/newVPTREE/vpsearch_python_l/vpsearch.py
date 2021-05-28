import os
import numpy as np
import cffi

ffi = cffi.FFI()
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "libvptree_multicore.so")
vpsearch = ffi.dlopen(path)

ffi.cdef("""void findNearestNeighbors(double* data, double* distances,
                                      int * indices, int dim, int num_rows,
                                      int num_neighbors, int num_threads);
                                      """)


ffi.cdef("""void findNearestNeighborsTarget(double* data, double* data_target, double* distances,
                                            int * indices, int dim, int num_rows,
                                            int num_rows_target,
                                            int num_neighbors, int num_threads);
                                            """)

def find_nearest_neighbors(data, num_neighbors, num_threads):
    rows, columns = data.shape

    data_copy = np.array(data, dtype=float, order='C', copy=True)
    indices =  np.zeros((rows, num_neighbors), dtype=np.int32)
    distances = np.zeros((rows, num_neighbors))

    cast_indices = ffi.cast('int*', indices.ctypes.data)
    cast_distances = ffi.cast('double*', distances.ctypes.data)
    cast_data = ffi.cast('double*', data_copy.ctypes.data)

    vpsearch.findNearestNeighbors(cast_data, cast_distances, cast_indices, \
                                        columns, rows, num_neighbors, num_threads)
    return indices, distances

def find_nearest_neighbors_target(data, data_target, num_neighbors, num_threads):
    rows, columns = data.shape
    target_rows, target_columns = data_target.shape

    if target_columns != columns:
        raise Exception('Data and data target cannot have different dimensionality.')

    data_copy = np.array(data, dtype=float, order='C', copy=True)
    data_target_copy = np.array(data_target, dtype=float, order='C', copy=True)
    indices =  np.zeros((target_rows, num_neighbors), dtype=np.int32)
    distances = np.zeros((target_rows, num_neighbors))

    cast_indices = ffi.cast('int*', indices.ctypes.data)
    cast_distances = ffi.cast('double*', distances.ctypes.data)
    cast_data = ffi.cast('double*', data_copy.ctypes.data)
    cast_target_copy = ffi.cast('double*', data_target_copy.ctypes.data)
    vpsearch.findNearestNeighborsTarget(cast_data, cast_target_copy, cast_distances, \
                                        cast_indices, columns, rows, target_rows, \
                                        num_neighbors, num_threads)

    return indices, distances
