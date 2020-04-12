//***************************************************************************
//  Broday Walker
//  Dr. Eduardo Colmenares
//  
//
//***************************************************************************

#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

const int GRID_X = 1; // Number of blocks in the grid on the x-axis 
const int BLOCK_X = 7; // Number of threads on the x-axis per block
const int BILLION = 1000000000; // Used as a substitute for the value of infinity
const int SIZE = 49; // Just for testing

__constant__ int adjMat_d[SIZE];

__global__ void dijkstra(int *adj_d, int *dist_d, int *parent_d, int s, int width)
{   
    printf("Adjacency matrix from the device\n");
    // Print the adjacency matrix for testing
    for(int i = 0; i < width; i++)
    {
        printf("Row %d: ", i);

        for(int k = 0; k < width; k++)
            printf("{%d} ", adj_d[i * width + k]);

        printf("\n");
    }
    printf("\n\n");
}

__global__ void printAdjMat(int *test, int width)
{
    if(threadIdx.x == 0)
        printf("Invoked\n");

    int tid = threadIdx.x;
    
    
    for(int i = 0; i < 7; i++)
        test[i * width + tid] = adjMat_d[i * width + tid] + 1;
}

int main()
{
    int num_edges, adj_size, dist_size, parent_size, start, end, width;
    // Host arrays
    int *adjMat, *dist, *parent;


    // Delete from here
    int *test_cpy, *test_cpy_d; //
    test_cpy = new int[49];     //
                                //
    for(int i = 0; i < 49; i++) //
        test_cpy[i] = 0;        //
    // Delete to here

    // Device arrays
    int *adj_d, *dist_d, *parent_d;

    // For error checking
    cudaError_t cudaErr;

    // Read in the number of vertices
    cin >> width;

    // This is a linearized adjacency matrix
    adjMat = new int[width * width];
    dist = new int[width];
    parent = new int[width];

    // Find size of arrays in bytes
    adj_size = width * width * sizeof(int);
    dist_size = width * sizeof(int); // Equal to the number of vertices
    parent_size = width * sizeof(int);

    // Fill arrays with 0s
    for(int i = 0; i < width * width; i++)
        adjMat[i] = 0;

    for(int i = 0; i < width; i++)
    {
        dist[i] = 0;
        parent[i] = BILLION;
    }

    // Manually set the starting and end vertices
    start = 1;
    dist[start] = 0; // Distance from the start vertex to the start vertex is always 0
    parent[start] = -1; // -1 is used to signal no parent
    end = 7;


    // Fill the adjacency matrix with data
    for(int i = 0; i < width; i++)
    {
        // Temporary storage for adjacent vertices and the weight of the edge
        vector<pair<int, int>> temp_edges;

        // Read in the number of adjacent vertices for the ith vertex
        cin >> num_edges;

        temp_edges.resize(num_edges);

        for(int j = 0; j < temp_edges.size(); j++)
            cin >> temp_edges[j].first >> temp_edges[j].second;

        // Add the adjacent vertices to the linearized adjacency-matrix for the ith vertex
        for(int j = 0; j < temp_edges.size(); j++)
            // This is basically blockIdx.x * blockDim.x + threadIdx.x where blockIdx.x 
            // corresponds with the iteration of the loop we are on
            adjMat[i * width + temp_edges[j].first] = temp_edges[j].second;

    }

    // Print weights for testing
    for(int i = 0; i < width; i++)
    {
        cout << "Row " << i << ": ";

        for(int j = 0; j < width; j++)
            cout << "{" << adjMat[i * width + j] << "} ";
        cout << '\n';
    }
    cout << "\n\n";
        
    // Copy the adjacency-matrix to constant memory on the device
    cudaErr = cudaMemcpyToSymbol(adjMat_d, adjMat, adj_size);
    if(cudaErr != cudaSuccess)
    {
        printf("Error copying from host to device symbol\n");
        return 1;
    }
    else
        printf("Successful copy from host to device symbol\n");
    
    
    
    // Delete from here

    cudaMalloc((void **)&test_cpy_d, adj_size);
    cudaMemcpy(test_cpy_d, test_cpy, adj_size, cudaMemcpyHostToDevice);

    dim3 dimGrid(GRID_X, 1);
    dim3 dimBlock(BLOCK_X, 1);

    printAdjMat<<<dimGrid, dimBlock>>>(test_cpy_d, 7);

    cudaMemcpy(test_cpy, test_cpy_d, adj_size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 7; i++)
    {
        for(int j = 0; j < 7; j++)
            printf("{%d} ", test_cpy[i * 7 + j]);

        printf("\n");
    }

    cudaFree(test_cpy_d);

    delete [] test_cpy;
    
    // Delete to here


    /*  COMMENTED OUT FOR TESTING  QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ

    // Allocate memory on the device and copy contents
    cudaMalloc((void **)&adj_d, adj_size);
    cudaMemcpy(adj_d, adjMat, adj_size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dist_d, dist_size);
    cudaMemcpy(dist_d, dist, dist_size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&parent_d, parent_size);
    cudaMemcpy(parent_d, parent, parent_size, cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 dimGrid(GRID_X, 1);
    dim3 dimBlock(BLOCK_X, 1);

    // Invoke the kernel
    dijkstra<<<dimGrid, dimBlock>>>(adj_d, dist_d, parent_d, start, width);

    // Copy the results back
    cudaMemcpy(dist, dist_d, dist_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(parent, parent_d, parent_size, cudaMemcpyDeviceToHost);

    // Print the distances from start vertex s
    for(int i = 0; i < width; i++)
        printf("%d to %d: %d\n", i, start, dist[i]);
    printf("\n\n");

    // Free the device memory
    cudaFree(adj_d);
    cudaFree(dist_d);
    cudaFree(parent_d);

    QQQQQQQQQQQQQQQQQQQQQQQQ   COMMENTED OUT FOR TESTING */

    // Free the host memory
    delete [] adjMat;
    delete [] dist;
    delete [] parent;

    return 0;
}