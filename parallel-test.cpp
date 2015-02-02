#include <iostream>

 // To compile these do g++ parallel-test.cpp -fopenmp
 using namespace std;

  int main()
  {
#pragma omp parallel
  {
    // Code inside this region runs in parallel. The number of times Hello is printed corresponds to the number of workers
    cout<<"Hello!\n";
  }

 #pragma omp parallel for
 for(int i = 0; i < 8; i++)
 {
    cout<<i;
 }
}
