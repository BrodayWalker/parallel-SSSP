//***************************************************************************
//  Broday Walker
//  Dr. Eduardo Colmenares
//   
//  Turing
//  Compilation:  /opt/cuda-9.0/bin/nvcc -arch=sm_37 -rdc=true dijkstra1B_shared.cu -o dijkstra1B_shared.exe
//  Execution: ./dijkstra_shared.exe < input.txt > output.txt
//***************************************************************************

#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <climits>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <stdio.h>

using namespace std;

__global__ void relax(int *dist_d, int *parent_d, int *visited_d, int *adjMat_d, int u, int width)
{
    int tid_x = threadIdx.x;
    int offset = u * width + tid_x;
 
    // Must use extern as the size of adjRow_ds is set by parent kernel
    extern __shared__ int adjRow_ds[];

    // Bring the adjacency row data for vertex u from global memory to shared memory
    adjRow_ds[tid_x] = adjMat_d[offset];
    __syncthreads();

    // Each thread will attempt to relax an adjacent vertex if an edge exists
    // If the vertex is unvisited and the edge exists
    // Self-loops are ignored
    if(tid_x != u && visited_d[tid_x] == 0 && adjRow_ds[tid_x] >= 0)
    {
        int v_dist = adjRow_ds[tid_x];

        if(dist_d[u] + v_dist < dist_d[tid_x])
        {
            // Relax  
            dist_d[tid_x] = dist_d[u] + v_dist;
            // Set parent
            parent_d[tid_x] = u;
        }
    }
}

__global__ void dijkstra(int *dist_d, int *parent_d, int *visited_d, int *adjMat_d, int s, int width)
{   
    dim3 childGrid(1, 1);
    dim3 childBlock(width, 1);

    int tid_x = threadIdx.x;
    // Used to dynamically allocate the size of shared memory for the child kernel in bytes
    int shared_mem_size = width * sizeof(int);
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
                relax<<<childGrid, childBlock, shared_mem_size>>>(dist_d, parent_d, visited_d, adjMat_d, u, width);
                cudaDeviceSynchronize();
            }
        }
    }
}

// A kernel for testing if the adjacency matrix was actually copied to the constant
// memory on the device
__global__ void transferAdjMat(int *test, int *adjMat_d, int width)
{
    int tid_x = threadIdx.x;
    
    for(int i = 0; i < width; i++)
        test[i * width + tid_x] = adjMat_d[i * width + tid_x];
}

int main()
{
    int vertices, cases = 1;
    
    // Declare the timer
    // Reference: 
    // https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);
    float milliseconds = 0;

	ofstream outfile;
	outfile.open("timings_GPU_1B_shared.txt");

    cin >> vertices;

    while(vertices != 0)
    {
        // Host Declarations
        int adj_size, dist_size, parent_size, visited_size, start, end, p;
        int *adjMat, *dist, *parent, *visited;
        stack<int> path;

        // Device declarations
        int *adjMat_d, *dist_d, *parent_d, *visited_d;

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

        /* Start Dijkstra */

        // Set distance of source vertex to 0;
        dist[start] = 0;
              
        // Set the dimensions of the grid and blocks
        dim3 gridDim(1, 1);
        dim3 blockDim(1, 1);

        // Allocate memory on the device 
        cudaMalloc((void **)&dist_d, dist_size);
        cudaMemcpy(dist_d, dist, dist_size, cudaMemcpyHostToDevice);

        cudaMalloc((void **)&parent_d, parent_size);
        cudaMemcpy(parent_d, parent, parent_size, cudaMemcpyHostToDevice);

        cudaMalloc((void **)&visited_d, visited_size);
        cudaMemcpy(visited_d, visited, visited_size, cudaMemcpyHostToDevice);

        cudaMalloc((void **)&adjMat_d, adj_size);
        cudaMemcpy(adjMat_d, adjMat, adj_size, cudaMemcpyHostToDevice);

        /* Record start time */
        cudaEventRecord(begin);

        // Invoke the kernel
        dijkstra<<<gridDim, blockDim>>>(dist_d, parent_d, visited_d, adjMat_d, start, vertices);

        /* Record end time */
        cudaEventRecord(stop);

        // Copy the results back
        cudaMemcpy(dist, dist_d, dist_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(parent, parent_d, parent_size, cudaMemcpyDeviceToHost);

        /* Block CPU execution until the specified event is recorded */
        cudaEventSynchronize(stop);
        /* Returns the elapsed time in milliseconds to the first argument */
        cudaEventElapsedTime(&milliseconds, begin, stop);

        // Free the device memory
        cudaFree(dist_d);
        cudaFree(parent_d);
        cudaFree(visited_d);

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

    // Print elapsed time
    outfile << "Elapsed time in milliseconds: " << milliseconds << '\n';
    

    outfile.close();
    return 0;
}