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
const int SIZE = 49; // The size of the adjacency matrix (WIDTH * WIDTH)
const int WIDTH = 7; // Number of vertices in the graph

// Hold the adjacency matrix in constant memory on the device as it will not be
// modified over the lifetime of the program
__constant__ int adjMat_d[SIZE];

__global__ void dijkstra_helper(int *dist_d, int *parent_d, int s, int width)
{   
    int tid_x = threadIdx.x;
    int block_x = blockIdx.x;
    int grid_x = dimGrid.x;




}

// A kernel for testing if the adjacency matrix was actually copied to the constant
// memory on the device
__global__ void printAdjMat(int *test, int width)
{
    int tid_x = threadIdx.x;
    
    for(int i = 0; i < width; i++)
        test[i * width + tid_x] = adjMat_d[i * width + tid_x];
}

int main()
{
    // Host Declarations
    int num_edges, adj_size, dist_size, parent_size, start, end, width;
    int *adjMat, *dist, *parent;
    
    // Priority queue
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

    // Device declarations
    int *adj_d, *dist_d, *parent_d;

    // For error checking
    cudaError_t cudaErr;

    width = WIDTH;

    // This is a linearized adjacency matrix
    adjMat = new int[width * width];
    dist = new int[width];
    parent = new int[width];

    // Find size of arrays in bytes
    adj_size = width * width * sizeof(int);
    dist_size = width * sizeof(int); // Equal to the number of vertices
    parent_size = width * sizeof(int);

    // Fill the adjacency-matrix with 0s
    for(int i = 0; i < width * width; i++)
        adjMat[i] = 0;

    // A vertex does not have a parent if its value is -1 (after running 
    // Dijkstra's algorithm, this will only be true for the starting vertex).
    for(int i = 0; i < width; i++)
    {
        dist[i] = BILLION;
        parent[i] = -1;
    }

    // Manually set the starting and end vertices
    start = 1;
    // Distance from the start vertex to the start vertex is always 0
    dist[start] = 0; 
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
    printf("Test print of all weights: \n");
    printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    for(int i = 0; i < width; i++)
    {
        cout << "Row " << i << ": ";

        for(int j = 0; j < width; j++)
            cout << "{" << adjMat[i * width + j] << "} ";
        cout << '\n';
    }
    printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
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
    
    // Set the dimensions of the grid and blocks
    dim3 dimGrid(GRID_X, 1);
    dim3 dimBlock(BLOCK_X, 1);




    // Allocate memory on the device 
    cudaMalloc((void **)&dist_d, dist_size);
    cudaMemcpy(dist_d, dist, dist_size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&parent_d, parent_size);
    cudaMemcpy(parent_d, parent, parent_size, cudaMemcpyHostToDevice);

    // Invoke the kernel
    dijkstra_helper<<<dimGrid, dimBlock>>>(dist_d, parent_d, start, width);

    // Copy the results back
    cudaMemcpy(dist, dist_d, dist_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(parent, parent_d, parent_size, cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(dist_d);
    cudaFree(parent_d);


    // Print the distances from start vertex s
    for(int i = 0; i < width; i++)
        printf("%d to %d: %d\n", i, start, dist[i]);
    printf("\n\n");


    // Free the host memory
    delete [] adjMat;
    delete [] dist;
    delete [] parent;

    return 0;
}