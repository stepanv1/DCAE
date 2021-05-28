#ifndef VPTREE_H
#define VPTREE_H

#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <queue>
#include <limits>
#include <omp.h>
#include <iostream>
#include <random>
#include <thread>
#include <cfloat>

#define NUM_THREADS(N) ((N) >= 0 ? (N) : omp_get_num_procs() + (N) + 1)

class DataPoint
{
    int _D;
    int _ind;
    double* _x;

public:
    DataPoint() {

        _D = 1;
        _ind = -1;
        _x = NULL;
    }
    DataPoint(int D, int ind, double* x) {

        _D = D;
        _ind = ind;
        _x = (double*) malloc(_D * sizeof(double));
		for (int d = 0; d < _D; d++) {
			_x[d] = x[d];
		}
    }
    DataPoint(const DataPoint& other) {

        if (this != &other) {
            _D = other.dimensionality();
            _ind = other.index();
            _x = (double*) malloc(_D * sizeof(double));
            for (int d = 0; d < _D; d++) _x[d] = other.x(d);
        }
    }
    ~DataPoint() {

      if (_x != NULL) free(this->_x);
    }
    DataPoint& operator= (const DataPoint& other) {         // asignment should free old object
        if (this != &other) {
            if (_x != NULL) free(_x);
            _D = other.dimensionality();
            _ind = other.index();
            _x = (double*) malloc(_D * sizeof(double));
            for (int d = 0; d < _D; d++) _x[d] = other.x(d);
        }
        return *this;
    }
    int index() const { return _ind; }
    int dimensionality() const { return _D; }
    double x(int d) const { return _x[d]; }
    void print_result() {
      for (int d = 0; d < this->_D; d++){
      }
    }
};


double euclidean_distance(const DataPoint &t1, const DataPoint &t2) {
    double dd = .0;
    for (int d = 0; d < t1.dimensionality(); d++) {
        dd += (t1.x(d) - t2.x(d)) * (t1.x(d) - t2.x(d));
    }
    return dd;
}

double euclidean_distance_print(const DataPoint &t1, const DataPoint &t2) {
    double dd = .0;
    for (int d = 0; d < t1.dimensionality(); d++) {
        std::cout << d << ": " << t1.x(d) << " - " << t2.x(d) << std::endl; 
        dd += (t1.x(d) - t2.x(d)) * (t1.x(d) - t2.x(d));
    }
    return dd;
}


template<typename T, double (*distance)( const T&, const T& )>
class VpTree
{
public:

    // Default constructor
    VpTree() : _root(0) {}

    // Destructor
    ~VpTree() {
        delete _root;
    }
	// Args: VPTree Pointer, Number of dimensions (int)


    // Function to create a new VpTree from data
    //void create(DataPoint** items) {
	void create(const std::vector<T>& items) {
        delete _root;
        _items = items;
        _root = buildFromPoints(0, items.size());
    }

    // Function that uses the tree to find the k nearest neighbors of target
    void search(const T& target, int k, DataPoint* results, double* distances)
    {
        unsigned int results_length = 0;
        unsigned int distances_length = 0;
        unsigned int max_length = k;
        // Use a priority queue to store intermediate results on

        std::priority_queue<HeapItem> heap;

        // Variable that tracks the distance to the farthest point in our results
        double tau = DBL_MAX;

        // Perform the searcg
        search(_root, target, k, heap, tau);

        // Gather final results
        //results->clear(); distances->clear();
        while (!heap.empty()) {

            if(results_length < max_length){

              results[results_length++] =  _items[heap.top().index];

            } else {

            }

            if(distances_length < max_length){
              distances[distances_length++] = heap.top().dist;
            } else {

            }

            heap.pop();
        }

        // Results are in reverse order
        for (int i = 0; i < (max_length / 2); i++) {
           DataPoint temporary = results[i];                 // temporary wasn't declared
           results[i] = results[(max_length - 1) - i];
           results[(max_length - 1) - i] = temporary;
        }

       for (int i = 0; i < (max_length / 2); i++) {
          double temporary = distances[i];                 // temporary wasn't declared
          distances[i] = distances[(max_length - 1) - i];
          distances[(max_length - 1) - i] = temporary;
       }
    }

private:
    //DataPoint* _items;
	std::vector<T> _items;

    // Single node of a VP tree (has a point and radius; left children are closer to point than the radius)
    struct Node
    {
        int index;              // index of point in node
        double threshold;       // radius(?)
        Node* left;             // points closer by than threshold
        Node* right;            // points farther away than threshold

        Node() :
            index(0), threshold(0.), left(0), right(0) {}

        ~Node() {               // destructor
            delete left;
            delete right;
        }
    }* _root;


    // An item on the intermediate result queue
    struct HeapItem {
        HeapItem( int index, double dist) :
            index(index), dist(dist) {}
        int index;
        double dist;
        bool operator<(const HeapItem& o) const {
            return dist < o.dist;
        }
    };

    // Distance comparator for use in std::nth_element
    struct DistanceComparator
    {
        const T& item;
        DistanceComparator(const T& item) : item(item) {}
        bool operator()(const T& a, const T& b) {
            return distance(item, a) < distance(item, b);
        }
    };

    // Function that (recursively) fills the tree
	Node* buildFromPoints(int lower, int upper)
	{
		if (upper == lower) {     // indicates that we're done here!
			return NULL;
		}

		// Lower index is center of current node
		Node* node = new Node();
		node->index = lower;

		if (upper - lower > 1) {      // if we did not arrive at leaf yet

									  // Choose an arbitrary point and move it to the start
			int i = (int)((double)rand() / RAND_MAX * (upper - lower - 1)) + lower;
			std::swap(_items[lower], _items[i]);

			// Partition around the median distance
			int median = (upper + lower) / 2;

			std::nth_element(_items.begin() + lower + 1,
				_items.begin() + median,
				_items.begin() + upper,
				DistanceComparator(_items[lower]));

			// Threshold of the new node will be the distance to the median
			node->threshold = distance(_items[lower], _items[median]);


			// Recursively build tree
			node->index = lower;
			node->left = buildFromPoints(lower + 1, median);
			node->right = buildFromPoints(median, upper);
		}

		// Return result
		return node;
	}
    // Helper function that searches the tree
    void search(Node* node, const T& target, int k, std::priority_queue<HeapItem>& heap, double& tau)
    {

        if (node == NULL) return;    // indicates that we're done here

        // Compute distance between target and current node
        double dist = distance(_items[node->index], target);

        // If current node within radius tau
        if (dist < tau) {
            if (heap.size() == k) heap.pop();                // remove furthest node from result list (if we already have k results)
            heap.push(HeapItem(node->index, dist));           // add current node to result list
            if (heap.size() == k) tau = heap.top().dist;    // update value of tau (farthest point in result list)
        }

        // Return if we arrived at a leaf
        if (node->left == NULL && node->right == NULL) {
            return;
        }

        // If the target lies within the radius of ball
        if (dist < node->threshold) {
            if (dist - tau <= node->threshold) {        // if there can still be neighbors inside the ball, recursively search left child first
                search(node->left, target, k, heap, tau);
            }

            if (dist + tau >= node->threshold) {        // if there can still be neighbors outside the ball, recursively search right child
                search(node->right, target, k, heap, tau);
            }

            // If the target lies outsize the radius of the ball
        } else {
            if (dist + tau >= node->threshold) {        // if there can still be neighbors outside the ball, recursively search right child first
                search(node->right, target, k, heap, tau);
            }

            if (dist - tau <= node->threshold) {         // if there can still be neighbors inside the ball, recursively search left child
                search(node->left, target, k, heap, tau);
            }
        }
    }
};


class VPSearchFunc {
public:
	//First function which creates VPTree. See function below for version that does not need pointer for points to be passed in.
	static void createEuclideanTree(double** data, VpTree<DataPoint, euclidean_distance>** _tree, std::vector<DataPoint> *_points, int num_rows, int dim, int num_threads)
	{
		//Set number of threads.
		omp_set_num_threads(NUM_THREADS(num_threads));
		//Initiate new tree at the passed in pointer.
		*_tree = new VpTree<DataPoint, euclidean_distance>();
		VpTree<DataPoint, euclidean_distance>* tree = *_tree;

		//Initiate vector for datapoints from the passed in points
		std::vector<DataPoint> points = *_points;
		
		//Assign data points
		#pragma omp parallel for
		for (int i = 0; i < num_rows; i++) {
			points[i] = DataPoint(dim, i, *data + (i * dim));
		}

		//Create the tree
		tree->create(points);
	}

	// First function without points pointer being passed in.
	static void createEuclideanTree(double** data, VpTree<DataPoint, euclidean_distance>** _tree, int num_rows, int dim, int num_threads)
	{
		omp_set_num_threads(NUM_THREADS(num_threads));
		*_tree = new VpTree<DataPoint, euclidean_distance>();

		VpTree<DataPoint, euclidean_distance>* tree = *_tree;
		std::vector<DataPoint> points(num_rows, DataPoint(dim, -1, *data));

		#pragma omp parallel for
		for (int i = 0; i < num_rows; i++) {
			points[i] = DataPoint(dim, i, *data + (i * dim));
		}

	
		tree->create(points);

		//De-allocate
		std::vector<DataPoint>().swap(points);
	}

	static DataPoint createDataPoint(double** data, int dim) {
		return DataPoint(dim, -1, *data);
	}



	static void findNearestNeighborsMatrix(double** _data, VpTree<DataPoint, euclidean_distance>** _tree, double** distances, int ** indices, int dim, int num_rows, int num_neighbors, int num_threads) {
		//Set the number of threads for parallelization
		omp_set_num_threads(NUM_THREADS(num_threads));

		double * data = *_data;

		//Declare points for target matrix.
		std::vector<DataPoint> points(num_rows, DataPoint(dim, -1, data));

		for (int i = 0; i < num_rows; i++) {
			points[i] = DataPoint(dim, i, data + (i * dim));
		}

		//Tree we are getting nearest neighbors from
		VpTree<DataPoint, euclidean_distance>* tree = *_tree;

		//Allocate space for distances and indices.
		*distances = (double*)calloc(num_rows * num_neighbors + 1, sizeof(double));
		double* distance_matrix = *distances;

		*indices = (int*)calloc(num_rows * num_neighbors + 1, sizeof(int));
		int* indices_matrix = *indices;

		//Matrix to keep track of general matrix indices as each sample has K nearest neighbors
		int * row = (int*)malloc((num_rows + 1) * sizeof(int));
		row[0] = 0;
		for (int n = 0; n < num_rows; n++) {
			row[n + 1] = row[n] + num_neighbors;
		}

		//DataPoint* indices_t = (DataPoint*)calloc((num_neighbors + 1) * omp_get_num_threads() * 2, sizeof(DataPoint));

		int steps_completed = 0;
		//int t_num = omp_get_num_threads();
		int t_num = num_threads;
		DataPoint ** indices_2d = new DataPoint*[t_num];
		for (int i = 0; i < t_num; i++) {
		  indices_2d[i] = new DataPoint[num_neighbors + 1];
		}
		double ** distances_2d = new double*[t_num];
		for (int i = 0; i < t_num; i++) {
		  distances_2d[i] = new double[num_neighbors + 1];
		}
		

		#pragma omp parallel num_threads(t_num)
		{
		  //printf("numthreads %d", t_num);
		  #pragma omp for
		  for (int n = 0; n < num_rows; n++) {
		    int tid = omp_get_thread_num();
		    //std::cout << n << std::endl;
		    //Declare local indices and local distances
		    //Call search on current point on the tree we pass in.
		    //std::cout << "Hello from thread " << tid << " / " << std::this_thread::get_id() << std::endl;
		    tree->search(points[n], num_neighbors + 1, indices_2d[tid], distances_2d[tid]);
				//std::cout << "Hello again from thread " << tid << " / " << std::this_thread::get_id() << std::endl;
				//	for (int i = 0; i < num_neighbors + 1; i++) {
				//		delete(indices[omp_get_thread_num() + i]);
				//	}
				//Increment the point we are at.

				//std::cout << n << std::endl;

				//Assign localized indices and matrices to the matrices passed in.
				for (int k = 0; k < num_neighbors; k++) {
							indices_matrix[row[n] + k] = indices_2d[tid][k + 1].index();
					    distance_matrix[row[n] + k] = distances_2d[tid][k + 1];
				}

				//De-allocate
					//delete[](local_indices);
				//delete[](local_distances);
			}
		}

		//De-allocate
		std::vector<DataPoint>().swap(points);
		delete row;
	}

	static void findNearestNeighborsMatrix(VpTree<DataPoint, euclidean_distance>** _tree, std::vector<DataPoint> *_points, double** distances, int ** indices, int dim, int num_rows, int num_neighbors, int num_threads) {
		omp_set_num_threads(NUM_THREADS(num_threads));
		std::vector<DataPoint> points = *_points;

		VpTree<DataPoint, euclidean_distance>* tree = *_tree;

		*distances = (double*)calloc(num_rows * num_neighbors, sizeof(double));
		double* distance_matrix = *distances;

		*indices = (int*)calloc(num_rows * num_neighbors, sizeof(int));
		int* indices_matrix = *indices;

		int * row = (int*)malloc((num_rows + 1) * sizeof(int));
		row[0] = 0;
		for (int n = 0; n < num_rows; n++) {
			row[n + 1] = row[n] + num_neighbors;
		}

		#pragma omp parallel for
		for (int n = 0; n < num_rows; n++) {
			DataPoint * local_indices;
			local_indices = new DataPoint[num_neighbors + 1];

			double * local_distances;
			local_distances = new double[num_neighbors + 1];
			tree->search(points[n], num_neighbors + 1, local_indices, local_distances);
			
			for (int k = 0; k < num_neighbors; k++) {
				indices_matrix[row[n] + k] = local_indices[k + 1].index();
				distance_matrix[row[n] + k] = local_distances[k + 1];
			}
			
		}
	}

	static void findNearestNeighborsMatrix(double** data, double** distances, int ** indices, int dim, int num_rows, int num_neighbors, int num_threads) {
		omp_set_num_threads(NUM_THREADS(num_threads));
		std::vector<DataPoint> points(num_rows, DataPoint(dim, -1, *data));

		for (int i = 0; i < num_rows; i++) {
			points[i] = DataPoint(dim, i, *data + (i * dim));
		}

		VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
		tree->create(points);

		*distances = (double*)calloc(num_rows * num_neighbors, sizeof(double));
		double* distance_matrix = *distances;

		*indices = (int*)calloc(num_rows * num_neighbors, sizeof(int));
		int* indices_matrix = *indices;

		int * row = (int*)malloc((num_rows + 1) * sizeof(int));
		row[0] = 0;
		for (int n = 0; n < num_rows; n++) {
			row[n + 1] = row[n] + num_neighbors;
		}

		#pragma omp parallel for
		for (int n = 0; n < num_rows; n++) {
			DataPoint * local_indices;
			local_indices = new DataPoint[num_neighbors + 1];

			double * local_distances;
			local_distances = new double[num_neighbors + 1];
			tree->search(points[n], num_neighbors + 1, local_indices, local_distances);
			for (int k = 0; k < num_neighbors; k++) {
				indices_matrix[row[n] + k] = local_indices[k + 1].index();
				distance_matrix[row[n] + k] = local_distances[k + 1];
			}
		}
	}

	//Second function
	static void findNearestNeighborsTarget(VpTree<DataPoint, euclidean_distance>** _tree, DataPoint target, double** distances, int ** indices, int dim, int num_rows, int num_neighbors) {
		VpTree<DataPoint, euclidean_distance>* tree = *_tree;

		*distances = (double*)calloc(num_neighbors, sizeof(double));
		double* distance_matrix = *distances;

		*indices = (int*)calloc(num_neighbors, sizeof(int));
		int* indices_matrix = *indices;

		int * row = (int*)malloc((num_rows + 1) * sizeof(int));

		DataPoint * local_indices;
		local_indices = new DataPoint[num_neighbors + 1];

		double * local_distances;
		local_distances = new double[num_neighbors + 1];
		tree->search(target, num_neighbors + 1, local_indices, local_distances);
		for (int k = 0; k < num_neighbors; k++) {
			indices_matrix[k] = local_indices[k + 1].index();
			distance_matrix[k] = local_distances[k + 1];
		}
	}


	static void findNearestNeighborsTarget(double** data, DataPoint target, double** distances, int ** indices, int dim, int num_rows, int num_neighbors) {
		std::vector<DataPoint> points(num_rows, DataPoint(dim, -1, *data));

		for (int i = 0; i < num_rows; i++) {
			points[i] = DataPoint(dim, i, *data + (i * dim));
		}

		VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
		tree->create(points);

		*distances = (double*)calloc(num_neighbors, sizeof(double));
		double* distance_matrix = *distances;

		*indices = (int*)calloc(num_neighbors, sizeof(int));
		int* indices_matrix = *indices;

		int * row = (int*)malloc((num_rows + 1) * sizeof(int));

		DataPoint * local_indices;
		local_indices = new DataPoint[num_neighbors + 1];

		double * local_distances;
		local_distances = new double[num_neighbors + 1];
		tree->search(target, num_neighbors + 1, local_indices, local_distances);
		for (int k = 0; k < num_neighbors; k++) {
			indices_matrix[k] = local_indices[k + 1].index();
			distance_matrix[k] = local_distances[k + 1];
		}
	}

	static void findNearestNeighborsTarget(double** data, double** data_target, double** distances, int ** indices, int dim, int num_rows, int num_rows_target, int num_neighbors, int num_threads) {
		omp_set_num_threads(NUM_THREADS(num_threads));

		std::vector<DataPoint> points(num_rows, DataPoint(dim, -1, *data));

		for (int i = 0; i < num_rows; i++) {
			points[i] = DataPoint(dim, i, *data + (i * dim));
		}

		std::vector<DataPoint> targets(num_rows, DataPoint(dim, -1, *data_target));

		for (int i = 0; i < num_rows_target; i++) {
			targets[i] = DataPoint(dim, i, *data_target + (i * dim));
		}

		VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
		tree->create(points);


		*distances = (double*)calloc(num_rows_target * num_neighbors, sizeof(double));
		double* distance_matrix = *distances;

		*indices = (int*)calloc(num_rows_target * num_neighbors, sizeof(int));
		int* indices_matrix = *indices;

		int * row = (int*)malloc((num_rows_target + 1) * sizeof(int));
		row[0] = 0;
		for (int n = 0; n < num_rows_target; n++) {
			row[n + 1] = row[n] + num_neighbors;
		}

		#pragma omp parallel for
		for (int n = 0; n < num_rows; n++) {
			DataPoint * local_indices;
			local_indices = new DataPoint[num_neighbors + 1];

			double * local_distances;
			local_distances = new double[num_neighbors + 1];
			tree->search(targets[n], num_neighbors + 1, local_indices, local_distances);
			for (int k = 0; k < num_neighbors; k++) {
				indices_matrix[row[n] + k] = local_indices[k + 1].index();
				distance_matrix[row[n] + k] = local_distances[k + 1];
			}
		}
	}

	static void generateRandomMatrix(double** data, int rows, int columns, double lower_bound, double upper_bound, int num_threads) {
		omp_set_num_threads(NUM_THREADS(num_threads));
		int num_datapoints = rows * columns;
		*data = (double*)calloc(num_datapoints+1, sizeof(double));
		double *random_matrix = *data;

		std::random_device randm;
		std::default_random_engine generator(randm());
		std::uniform_real_distribution<double> distribution(lower_bound, upper_bound);

		#pragma omp parallel for
		for (int i = 0; i < num_datapoints; i++) {
				
			double random_number = distribution(generator);

			random_matrix[i] = random_number;
		}
		std::cout << std::endl;
	}


	static void printMatrix(double ** data, int rows, int columns)
	{
		double *data_matrix = *data;
		for (int i = 0; i < rows; i++) {
			std::cout << i << "| ";
			for (int k = 0; k < columns; k++) {
				std::cout << data_matrix[i*columns + k] << " ";
			}
			std::cout << std::endl;
		}
	}

	static void printMatrix(int ** data, int rows, int columns)
	{
		int *data_matrix = *data;
		for (int i = 0; i < rows; i++) {
			std::cout << i << "| ";
			for (int k = 0; k < columns; k++) {
				std::cout << data_matrix[i*columns + k] << " ";
			}
			std::cout << std::endl;
		}
	}
};

#endif
 
