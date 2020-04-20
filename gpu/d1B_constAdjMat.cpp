//***************************************************************************
//  Broday Walker
//  Dr. Eduardo Colmenares
//
//  This program is for testing a parallel implementation of Dijkstra's 
//  algorithm for one specific input graph that has 7 vertices.
//
//   
//  Turing
//  Compilation:  /opt/bin/cuda-9.0/bin/nvcc -arch=sm_35 -rdc=true d1BconstAdjMat.cu -o d1B.exe
//  Execution: ./d1B.exe < input.txt > output.txt
//***************************************************************************

#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stack>

using namespace std;

const int GRID_X = 1; // Number of blocks in the grid on the x-axis 
const int BLOCK_X = 7; // Number of threads on the x-axis per block
const int SIZE = 49;

// Hold the adjacency matrix in constant memory on the device as it will not be
// modified over the lifetime of the program
__constant__ int adjMat_d[SIZE];


__global__ void relax(int *dist_d, int *parent_d, int *visited_d, int u, int width)
{
    int tid_x = threadIdx.x;

    int offset = u * width + tid_x;

    // Each thread will attempt to relax an adjacent vertex if an edge exists
    // If the vertex is unvisited and the edge exists
    if(tid_x != u && visited_d[tid_x] == 0 && adjMat_d[offset] >= 0)
    {
        int v_dist = adjMat_d[offset];

        if(dist_d[u] + v_dist < dist_d[tid_x])
        {
            // Relax  
            dist_d[tid_x] = dist_d[u] + v_dist;
            // Set parent
            parent_d[tid_x] = u;
        }
    }
    
}

__global__ void dijkstra(int *dist_d, int *parent_d, int *visited_d, int s, int width)
{   
    dim3 childGrid(GRID_X, 1);
    dim3 childBlock(BLOCK_X, 1);

    int tid_x = threadIdx.x;

    bool all_visited = false;

    if(tid_x == 0)
    {
        while(!all_visited)
        {
            all_visited = true;
            bool next_found = false;

            // These are u sed as the variables for the next vertex's index and minimum cost
            int u, u_min, index = 0;

            // Find the next unvisited vertex (if there is one)
            while (!next_found && index < width)
            {
                if(visited_d[index] == 0)
                {
                    u = index;
                    u_min = dist_d[index];
                    all_visited = false;
                    next_found = true;
                }

                index++;
            }

            if(!all_visited)
            {
                // Find shortest distance of all the unvisited vertices
                // Start at the first unvisited vertex encountered in the previous loop to 
                // try to save some computation.
                int i = u + 1;
                while(i < width)
                {
                    if(visited_d[i] == 0 && dist_d[i] < u_min)
                    {
                        u = i;
                        u_min = dist_d[i];
                    }

                    i++;
                }

                // Now we have the next vertex to process in u
                // Set u as visited
                visited_d[u] = 1;

                // Attempt to relax all vertices adjacent to vertex u
                relax<<<childGrid, childBlock>>>(dist_d, parent_d, visited_d, u, width);
                cudaDeviceSynchronize();
            }
        }
    }
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
    int vertices, cases = 1;

    cin >> vertices;

    while(vertices != 0)
    {
        // Host Declarations
        int adj_size, dist_size, parent_size, visited_size, start, end, p;
        int *adjMat, *dist, *parent, *visited;
        stack<int> path;

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
            adjMat[i] = -1;

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
            int num_edges, u, w;

            // Read in the number of adjacent vertices for the ith vertex
            cin >> num_edges;

            // Add the adjacent vertices to the linearized adjacency-matrix for the ith vertex
            for(int j = 0; j < num_edges; j++)
            {
                cin >> u >> w;
                int offset = i * vertices + u;

                if(adjMat[offset] == -1 || w < adjMat[offset])
                    // This is basically blockIdx.x * blockDim.x + threadIdx.x where blockIdx.x 
                    // corresponds with the iteration of the loop we are on
                    adjMat[offset] = w;
            }
        }

        cin >> start >> end;

        // Set distance of source vertex to 0;
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
        
        // Set the dimensions of the grid and blocks
        dim3 gridDim(1, 1);
        dim3 blockDim(1, 1);


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

        // Start at the end vertex and work back through the parent vertices to build
        // the path
        p = end;
		path.push(p);
		while (p != start)
		{
			p = parent[p];
			path.push(p);
		}

		// Print result
		cout << "Case " << cases << ": Path =";

		while (!path.empty())
		{
            // NOTE
            // Add one to the vertex ID number if comparing results to UVA problem 341 
			cout << " " << path.top() + 1;
			path.pop();
		}

		cout << "; " << dist[end] << " second delay\n";


        // Free the host memory
        delete [] adjMat;
        delete [] dist;
        delete [] parent;
        delete [] visited;


        cases++;
        cin >> vertices;

    }

    return 0;
}