//***************************************************************************
//  Broday Walker
//
//  This program solves UVA341 using a compressed sparse row representation
//  of the input graph. Instead of compressing 0s, this implementation
//  compresses values of -1, as that is the default value used for an edge
//  weight that does not exist. Using -1 as the default edge weight value
//  allows the algorithm to correctly process edge weights of 0, which occur
//  in one of the testing datasets for UVA341.
//
//***************************************************************************

#include <iostream>
#include <vector>
#include <queue>
#include <stack>

using namespace std;

int main()
{
    // Declarations
    int num_inters, num_edges, v, delay, start, end, p, cases = 1;

    // Read in number of intersections
    cin >> num_inters;
    while(num_inters != 0)
    {
        // Distance vector:
        // 0th element not used due to intersection numbering scheme
        // INT_MAX is used as infinity
        vector<int> dist(num_inters + 1, INT_MAX); 

        // Parent vector
        vector<int> parent(num_inters + 1, -1);

        // Path queue
        vector<int> path;

        // A beautiful vector of vectors of integers
        // Must add 1 to the size of the vector because intersections are
        // numbered from 1 to num_inters
        vector<vector<int>> adjMat(num_inters + 1, vector<int>(num_inters + 1, -1));

        // Priority queue
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

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

        // Build the compressed spare row representation

        // weights and col_index will be the same size
        // row_ptr is size num_inters + 2 for easy processing in the algorithm
        // we add 1 to the row_ptr size because vertices are numbered from 1 to n
        // we add another 1 to the row_ptr size to hold the end element of the last row

        vector<int> weights, col_index, row_ptr(num_inters + 2, 0);

        // Process the ith row
        for(int i = 0; i <= num_inters; i++)
        {
            int offset = 0;

            // Process all column elements in the ith row
            for(int j = 0; j <= num_inters; j++)
            {
                // If the data is not a -1
                if(adjMat[i][j] != -1)
                {
                    // Insert the data
                    weights.push_back(adjMat[i][j]);
                    // Keep track of the data's original column location
                    col_index.push_back(j);
                    // offset will hold the number of values that are not -1
                    offset++;

                }
            }

            // Calculate the index + 1 of the last element in row i
            // Note that this "end" of the row is actually the start of the next row
            row_ptr[i + 1] = row_ptr[i] + offset; 
        }

        // Read in the start and end vertices
        cin >> start >> end;


        // Dijkstra's

        // Set start vertex's distance from start to 0
        dist[start] = 0;

        // Because pq is a priority queue, the first int is the distance from the start vertex
        // while the second int is the vertex number (id)
        pq.push(pair<int, int>(0, start));

        while(!pq.empty())
        {
            // Get next vertex
            pair<int, int> front = pq.top();
            pq.pop();

            int d = front.first; // Distance from start
            int u = front.second; // vertex id number
            int row_start, row_end;

            // This check is necessary as this implementation of Dijkstra's does a lazy delete
            // In other words, don't process the path if it has already been relaxed
            if (d > dist[u])
                continue;

            // Calculate vertex u's row
            row_start = row_ptr[u];
            row_end = row_ptr[u + 1];

            // Iterate through vertex u's adjacency matrix relaxing edges 
            for (int j = row_start; j < row_end; j++)
            {
                pair<int, int> v = pair<int, int>(col_index[j], weights[j]);

                if (dist[u] + v.second < dist[v.first])
                {
                    // Relax
                    dist[v.first] = dist[u] + v.second;
                    // Set parent
                    parent[v.first] = u;
                    // Enqueue
                    pq.push(pair<int, int>(dist[v.first], v.first));
                }
            }
        }
        
        // Start at this vertex and work back through the parent vertices to the start
        p = end;
        path.push_back(p);
        while (p != start)
        {
            path.push_back(parent[p]);
            p = parent[p];
        }
        
        // Print result
        cout << "Case " << cases << ": Path =";
        for(int i = path.size() - 1; i >= 0; i--)
            cout << " " << path[i];
        cout << "; " << dist[end] << " second delay\n";
        


        // Increment the case being processed
        cases++;
        // Read in number of intersections
        cin >> num_inters;
    }

    return 0;
}