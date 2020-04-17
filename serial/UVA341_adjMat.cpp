// ************************************************************************ //
//                             Broday Walker
//                                UVA 341
//  Description: Solved using Dijkstra's Algorithm
//
//  Compilation: g++ UVA341.cpp -o main.exe -std=c++11
// *********************************************************************** //

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <queue>

using namespace std;

const int BILLION = 1000000000;

int main()
{
    // Open the input file
    ifstream infile;
    infile.open("input.txt");

    // Declarations
    int num_inters, num_edges, v, delay, start, end, p, cases = 1;

    // Read in number of intersections
    cin >> num_inters;
    while(num_inters != 0)
    {
        // Distance vector:
        // 0th element not used due to intersection numbering scheme
        // BILLION is used as infinity
        vector<int> dist(num_inters + 1, BILLION); 

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

        // Read in the start and end nodes
        cin >> start >> end;

        // Dijkstra's

        // Set start node's distance from start to 0
        dist[start] = 0;

        // Because pq is a priority queue, the first int is the distance from the start node
        // while the second int is the node number (id)
        pq.push(pair<int, int>(0, start));

        while(!pq.empty())
        {
            // Get next node
            pair<int, int> front = pq.top();
            pq.pop();

            int d = front.first; // Distance from start
            int u = front.second; // Node id number

            // This check is necessary as this implementation of Dijkstra's does a lazy delete
            // In other words, don't process the path if it has already been relaxed
            if (d > dist[u])
                continue;

            // Iterate through node u's adjacency matrix relaxing edges 
            for (int j = 0; j < adjMat[u].size(); j++)
            {
                // Get next adjacent vertex in the adjacency matrix
                if(adjMat[u][j] >= 0)
                {
                    pair<int, int> v = pair<int, int>(j, adjMat[u][j]);

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
        }
        
        // Start at this node and work back through the parent nodes to the start
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




    // Close the input file
    infile.close();
    return 0;
}