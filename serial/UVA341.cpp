// ************************************************************************ //
//                             Broday Walker
//                                UVA 341
//  Description: Solved using Dijkstra's Algorithm
//
//  Compilation: g++ UVA341.cpp -o main.exe -std=c++11
// *********************************************************************** //

#include <iostream>
#include <fstream>
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
    int num_inters, num_edges, edge, delay, start, end, p, cases = 1;

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

        // A beautiful vector of vectors of pairs of integers
        vector<vector<pair<int,int>>> adjList;
        // Must add 1 to the size of the vector because intersections are
        // numbered from 1 to num_inters
        adjList.resize(num_inters + 1);

        // Priority queue
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

        // Build the adjacency list
        for (int i = 1; i <= num_inters; i++)
        {
            // Read in the number of edges from this node to other nodes
            cin >> num_edges;
            // Build this node's adjacency list
            for (int j = 0; j < num_edges; j++)
            {
                cin >> edge >> delay;
                adjList[i].push_back(pair<int, int>(edge, delay));
            }
        }
        // Read in the start and end nodes
        cin >> start >> end;

        /* Dijkstra's */

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

            // Iterate through node u's adjacency list relaxing edges 
            for (int j = 0; j < adjList[u].size(); j++)
            {
                // Get next node in the adjacency list
                pair<int, int> v = adjList[u][j];

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