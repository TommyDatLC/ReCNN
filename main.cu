#include <iostream>


#include "Component/TommyDatMatrix.h"
using namespace std;
using namespace TommyDat;
#include "Component/Kernel3D.h"
int main() {
    Matrix a = Matrix(10,10 ,2.0f);
    Matrix b = Matrix(10,10 ,2.0f);
    b.set(3,4,-14);
    //kernel.set(1,1,4);
    Kernel3D<float> test(3,3,3);
    cout  << test;
}
