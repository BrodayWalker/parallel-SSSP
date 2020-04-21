//***************************************************************************
//	Broday Walker
//
// 	This is UVA341 solved without the use of a priority queue in the 
// 	implementation of Dijkstra's algorithm.
//
//	g++ UVA341_noQ.cpp -o noQ.exe -std=c++11
//	./main.exe < input.txt > output.txt
//***************************************************************************
#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <climits>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

int main()
{
	// Declarations
	int num_inters, num_edges, v, delay, start, end, p, cases = 1;
	double diff;
	struct timespec begin, stop;

	ofstream outfile;
	outfile.open("timings_UVA341_noQ.txt");

	// Start timer
	clock_gettime(CLOCK_MONOTONIC, &begin);

	// Read in number of intersections
	cin >> num_inters;
	while (num_inters != 0)
	{
		bool all_visited = false;

		// Distance vector:
		// 0th element not used due to intersection numbering scheme
		// INT_MAX is used as infinity
		vector<int> dist(num_inters + 1, INT_MAX);

		// Parent vector
		vector<int> parent(num_inters + 1, -1);

		// Visited vector
		vector<int> visited(num_inters + 1, 0);

		// Path stack
		stack<int> path;

		// A beautiful vector of vectors of integers
		// Must add 1 to the size of the vector because intersections are
		// numbered from 1 to num_inters
		vector<vector<int>> adjMat(num_inters + 1, vector<int>(num_inters + 1, -1));

		// Build the adjacency matrix
		for (int i = 1; i <= num_inters; i++)
		{
			// Read in the number of edges from this vertex to other vertices
			cin >> num_edges;
			// Build this vertex's adjacency list
			for (int j = 0; j < num_edges; j++)
			{
				cin >> v >> delay;

				// This check is necessary to ensure only the shortest edge between two
				// vertices is added to the adjacency-matrix representation of the input 
				// graph. This design decision results in the removal of any multiples of
				// a direct edge between two vertices. In other words, only the shortest
				// direct edge between vertices is recorded. The adjacency-list version
				// of the UVA341 solution does keep the extra paths.
				if (adjMat[i][v] == -1 || delay < adjMat[i][v])
					adjMat[i][v] = delay;
			}
		}

		// Read in the start and end vertices
		cin >> start >> end;

		// Dijkstra's

		// Set start vertex's distance from start to 0
		dist[start] = 0;

		while (!all_visited)
		{
			all_visited = true;
			bool next_found = false;
			// These are used as the variables for the next vertex's 
			// index and minimum cost
			int u, u_min;
			int index = 1;

			// Find next unvisited vertex (if there is one)
			while (!next_found && index <= num_inters)
			{
				if (visited[index] == 0)
				{
					u = index;
					u_min = dist[index];
					all_visited = false;
					next_found = true;
				}

				index++;
			}

			if (!all_visited)
			{
				// Find shortest distance of all the unvisited vertices
				// Start at the first unvisited vertex encountered
				// in the previous loop to try to save some computation.
				int i = u + 1;
				while (i <= num_inters)
				{
					if (visited[i] == 0 && dist[i] < u_min)
					{
						u = i;
						u_min = dist[i];
					}

					i++;
				}
			

				// Now we have the next vertex to process in u_index
				// Set vertex u as visited
				visited[u] = 1;

				// Iterate through vertex u's adjacency matrix, relaxing edges 
				for (int j = 0; j < adjMat[u].size(); j++)
				{
					// Get next adjacent vertex in the adjacency matrix if the
					// vertex is unvisited and it has a valid weight
					if (visited[j] == 0 && adjMat[u][j] >= 0)
					{
						int v = j;
						int v_dist = adjMat[u][v];

						if (dist[u] + v_dist < dist[v])
						{
							// Relax
							dist[v] = dist[u] + v_dist;
							// Set parent
							parent[v] = u;
						}
					}
				}
			}
		}

		// Start at this node and work back through the parent nodes to the start
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
			cout << " " << path.top();
			path.pop();
		}

		cout << "; " << dist[end] << " second delay\n";

		// Increment the case being processed
		cases++;
		// Read in number of intersections
		cin >> num_inters;
	}

	// Get end time
	clock_gettime(CLOCK_MONOTONIC, &stop);

	// Calculate elapsed time in milliseconds
	diff = (stop.tv_sec - begin.tv_sec) + (stop.tv_nsec - begin.tv_nsec) / 1000000.0;

	outfile << "Time elapsed is " << diff << " milliseconds.\n";

	outfile.close();
	return 0;
}