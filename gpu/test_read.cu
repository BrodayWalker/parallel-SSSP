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
const int SIZE = 49;

// Hold the adjacency matrix in constant memory on the device as it will not be
// modified over the lifetime of the program
__constant__ int adjMat_d[SIZE];

__global__ void dijkstra(int *dist_d, int *parent_d, int *visited_d, int s, int width)
{   
    int tid_x = threadIdx.x;
    int block_x = blockIdx.x;
    int grid_x = gridDim.x;




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
    int vertices;

    cin >> vertices;

    while(vertices != 0)
    {
        // Host Declarations
        int num_edges, adj_size, dist_size, parent_size, visited_size, start, end;
        int *adjMat, *dist, *parent, *visited;

        // Device declarations
        int *adj_d, *dist_d, *parent_d, *visited_d;

        // For error checking
        cudaError_t cudaErr;

        // This is a linearized adjacency matrix
        adjMat = new int[vertices * vertices];
        dist = new int[vertices];
        parent = new int[vertices];
        visited = new int[vertices];

        // Find size of arrays in bytes
        adj_size = vertices * vertices * sizeof(int);
        dist_size = vertices * sizeof(int); // Equal to the number of vertices
        parent_size = vertices * sizeof(int);
        visited_size = vertices * sizeof(int);

        // Fill the adjacency-matrix with 0s
        for(int i = 0; i < vertices * vertices; i++)
            adjMat[i] = 0;

        // A vertex does not have a parent if its value is -1 (after running 
        // Dijkstra's algorithm, this will only be true for the starting vertex).
        for(int i = 0; i < vertices; i++)
        {
            dist[i] = INT_MAX;
            parent[i] = -1;
            visited[i] = 0;
        }

        // Fill the adjacency matrix with data
        for(int i = 0; i < vertices; i++)
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
                adjMat[i * vertices + temp_edges[j].first] = temp_edges[j].second;

        }

        cin >> start >> end;
        dist[start] = 0;

        // Print weights for testing
        printf("Test print of all weights: \n");
        printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        for(int i = 0; i < vertices; i++)
        {
            cout << "Row " << i << ": ";

            for(int j = 0; j < vertices; j++)
                cout << "{" << adjMat[i * vertices + j] << "} ";
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
        dim3 gridDim(GRID_X, 1);
        dim3 blockDim(BLOCK_X, 1);




        // Allocate memory on the device 
        cudaMalloc((void **)&dist_d, dist_size);
        cudaMemcpy(dist_d, dist, dist_size, cudaMemcpyHostToDevice);

        cudaMalloc((void **)&parent_d, parent_size);
        cudaMemcpy(parent_d, parent, parent_size, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&visited_d, visited_size);
        cudaMemcpy(visited_d, visited, visited_size, cudaMemcpyHostToDevice);

        // Invoke the kernel
        dijkstra<<<gridDim, blockDim>>>(dist_d, parent_d, visited_d, start, vertices);

        // Copy the results back
        cudaMemcpy(dist, dist_d, dist_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(parent, parent_d, parent_size, cudaMemcpyDeviceToHost);

        // Free the device memory
        cudaFree(dist_d);
        cudaFree(parent_d);
        cudaFree(visited_d);


        // Print the distances from start vertex s
        for(int i = 0; i < vertices; i++)
            printf("%d to %d: %d\n", i, start, dist[i]);
        printf("\n\n");

        // Print the parent array
        for(int i = 0; i < vertices; i++)
            printf("%d: %d\n", i, parent[i]);
        printf("\n\n");


        // Free the host memory
        delete [] adjMat;
        delete [] dist;
        delete [] parent;
        delete [] visited;


        cin >> vertices;
    }

    return 0;
}