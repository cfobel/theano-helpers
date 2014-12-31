#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "test.hpp"


int main(int argc, char const* argv[]) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0] << " <N> scalar a b" << std::endl
      << std::endl;
    std::cerr << "computes: " << std::endl << "    (scalar * a) / b"
      << std::endl;
    exit(-1);
  }

  typedef int A_T;
  typedef float B_T;
  typedef B_T output_T;

  const int N = atoi(argv[1]);
  const output_T scalar = atof(argv[2]);
  const A_T a = atof(argv[3]);
  const B_T b = atof(argv[4]);

  typedef typename thrust::device_vector<A_T>::iterator A_iterator;
  typedef typename thrust::device_vector<B_T>::iterator B_iterator;

  thrust::device_vector<A_T> A(N);
  thrust::device_vector<B_T> B(N);
  thrust::device_vector<output_T> C(N);

  thrust::fill(A.begin(), A.end(), a);
  thrust::fill(B.begin(), B.end(), b);

  uuid00::Run()(scalar, A.begin(), N, B.begin(), C.begin());
  uuid00::Run().other(scalar, A.begin(), B.begin());

  for (int i = 0; i < N; i++) {
    std::cout << std::setw(4) << C[i];
  }
  std::cout << std::endl;

  return 0;
}
