//***************************************************************************
//  Broday Walker
//
//  This program generates a fully connected graph using the same format as
//  UVA problem 341.
//
//  Compilation: g++ full_conn_gen_UVA.cpp -o full_conn_gen_UVA.exe -std=c++11
//  Execution: ./full_conn_gen_UVA.exe > some_file.txt
//***************************************************************************
#include <iostream>
#include <time.h>

using namespace std;

const int VERTICES = 4096;
const int WEIGHT_LIMIT = 101;

int main()
{
    // Seed the random number generator
    srand(time(NULL));

    cout << VERTICES << '\n';

    for(int i = 1; i <= VERTICES; i++)
    {
        cout << VERTICES << " ";

        for(int j = 1; j <= VERTICES; j++)
            cout << (rand() % VERTICES) + 1 << " " << (rand() % WEIGHT_LIMIT) + 1 << " ";

        cout << '\n';
    }

    // Pick random source and end vertices
    cout << (rand() % VERTICES) + 1 << " " << (rand() % VERTICES) + 1 << '\n';
    cout << 0 << '\n';

    return 0;
}