//***************************************************************************
//  Broday Walker
//  Dr. Eduardo Colmenares
//   
//  Turing
//  Note: -rdc=true must be used as CUDA dynamic parallelism requires separate 
//  compilation and linking
//  Compilation:  /opt/cuda-9.0/bin/nvcc -arch=sm_37 -rdc=true dijkstra1B.cu -o dijkstra1B.exe
//  Execution: ./dijkstra1B.exe < ~/5433/dijkstra/graphs/t_thou.txt > thou_output.txt
//
//  Maverick2 GTX queue (use submission script)
//  Compilation: nvcc dijkstra1B.cu -o dijkstra1B.exe -std=c++11 -O3 -arch=compute_61 -code=sm_61 -rdc=true
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

const float ONE_MILLION = 1000000;

timespec elapsed(timespec, timespec);

__global__ void relax(int *dist_d, int *parent_d, int *visited_d, int *adjMat_d, int u, int width)
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

__global__ void dijkstra(int *dist_d, int *parent_d, int *visited_d, int *adjMat_d, int s, int width)
{   
    dim3 childGrid(1, 1);
    dim3 childBlock(width, 1);

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
                relax<<<childGrid, childBlock>>>(dist_d, parent_d, visited_d, adjMat_d, u, width);
                cudaDeviceSynchronize();
            }
        }
    }
}

// A kernel for testing if the adjacency matrix was actually copied to the constant
// memory on the device
__global__ void printAdjMat(int *test, int *adjMat_d, int width)
{
    int tid_x = threadIdx.x;
    
    for(int i = 0; i < width; i++)
        test[i * width + tid_x] = adjMat_d[i * width + tid_x];
}

int main()
{
    int vertices, cases = 1;
	struct timespec begin, stop, diff;

	ofstream outfile;
	outfile.open("timings_GPU_1B_global.txt");

	// Start timer
	clock_gettime(CLOCK_MONOTONIC, &begin);

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

        // Invoke the kernel
        dijkstra<<<gridDim, blockDim>>>(dist_d, parent_d, visited_d, adjMat_d, start, vertices);

        // Copy the results back
        cudaMemcpy(dist, dist_d, dist_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(parent, parent_d, parent_size, cudaMemcpyDeviceToHost);

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

    // Get end time
	clock_gettime(CLOCK_MONOTONIC, &stop);

	// Calculate elapsed time in nanoseconds
	diff = elapsed(begin, stop);

	outfile << "Time elapsed is " << diff.tv_sec << " seconds and " << diff.tv_nsec / ONE_MILLION << " milliseconds.\n";

    outfile.close();

    return 0;
}

// Special thanks to Guy Rutenberg for this hint on negative timing values were being reported
// https://www.guyrutenberg.com/2007/09/22/profiling-code-using-clock_gettime/comment-page-1/
timespec elapsed(timespec start, timespec end)
{
	timespec temp;

	if ((end.tv_nsec - start.tv_nsec) < 0) 
    {
		temp.tv_sec = end.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
	} 
    else 
    {
		temp.tv_sec = end.tv_sec - start.tv_sec;
		temp.tv_nsec = end.tv_nsec - start.tv_nsec;
	}

	return temp;
}