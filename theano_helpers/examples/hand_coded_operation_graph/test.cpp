#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "test.hpp"


int main(int argc, char const* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <N>" << std::endl;
    exit(-1);
  }

  const int N = atoi(argv[1]);

  typedef int A_T;
  typedef float B_T;
  typedef B_T output_T;

  typedef typename thrust::device_vector<A_T>::iterator A_iterator;
  typedef typename thrust::device_vector<B_T>::iterator B_iterator;

  thrust::device_vector<A_T> A(N);
  thrust::device_vector<B_T> B(N);
  thrust::device_vector<output_T> C(N);

  thrust::fill(A.begin(), A.end(), 1);
  thrust::fill(B.begin(), B.end(), 2.5);

  uuid00::Run()(A.begin(), N, B.begin(), C.begin());

  for (int i = 0; i < N; i++) {
    std::cout << std::setw(4) << C[i];
  }
  std::cout << std::endl;

  return 0;
}
