#include <iostream>


#include "Component/TommyDatMatrix.h"
using namespace std;
using namespace TommyDat;

int main() {
    Matrix a = Matrix(14,14 ,2.0f);
    Matrix kernel = Matrix(3,3,1.f);
    //kernel.set(1,1,4);

    cout  << a.convolution(kernel) << kernel;
}
