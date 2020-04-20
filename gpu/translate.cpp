//***************************************************************************
//  Broday Walker
//  This program decrements all vertex IDs for UVA341 by 1. The new numbering
//  system numbers vertices from 0 to N - 1 rather than 1 to N.
//
//***************************************************************************
#include <iostream>

using namespace std;

int main()
{
    int vertices, adjacent, u, weight, start, end;

    cin >> vertices;

    while(vertices != 0)
    {
        cout << vertices << '\n';

        for(int i = 0; i < vertices; i++)
        {
            cin >> adjacent;
            cout << adjacent << " ";

            for(int j = 0; j < adjacent; j++)
            {
                cin >> u >> weight;
                cout << u - 1 << " " << weight << " ";
            }

            cout << '\n';
        }

        cin >> start >> end;
        cout << start - 1 << " " << end - 1 << '\n';

        cin >> vertices;
    }

    cout << vertices << '\n';

    return 0;
}