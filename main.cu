#include <iostream>


#include "Component/TommyDatMatrix.h"
using namespace std;
using namespace TommyDat;

int main() {
    Matrix a = Matrix(3,3,6.0f);
    Matrix b = Matrix(3,3,2.0f);
    a.set(1,1,23.0f);
    Matrix d =  a - b;
    cout << d;
}
