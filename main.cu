#include <iostream>

#include "Utils/GPUmax.h"
#include "Utils/GPUPrefixSum.h"
#include "Utils/Utility.cuh"
using namespace std;
int main() {
    float a[6] = {1,2,3,4,5,6};
    int lenA = sizeof(a) / sizeof(float);
    // cout << GPUmax(a,lenA);
    CallPrefixSum(a,lenA);
    for (int i = 0;i < lenA;i++) {
        cout << a[i] << ' ';
    }
}