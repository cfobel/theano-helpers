#ifndef ___TEST__HPP___
#define ___TEST__HPP___

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>
#include "Transform.hpp"


namespace uuid00 {
  struct Run {
    template <typename A, typename B, typename C>
    void operator() (
      typename thrust::iterator_traits<C>::value_type scalar,
      A a, size_t N, B b, C c) {
      /* Operation:
       *
       *     c = (scalar * a) / b
       */
      typedef typename thrust::iterator_traits<C>::value_type result_type1;
      typedef thrust::divides<result_type1> Op1;

      typedef typename thrust::iterator_traits<B>::value_type result_type0;
      typedef thrust::multiplies<result_type0> Op0;

      typedef theano_helpers::Transform2
        <Op0, thrust::constant_iterator<result_type0>, A> transform0;
      typedef typename transform0::iterator iterator0;

      typedef theano_helpers::Transform2<Op1, iterator0, B> transform1;
      typedef typename transform1::iterator iterator1;

      iterator0 node0 = transform0(thrust::constant_iterator<result_type0>(scalar), a, Op0()).begin();
      iterator1 node1 = transform1(node0, b, Op1()).begin();

      thrust::copy_n(node1, N, c);
    }
  };

}  // end namespace

#endif  // #ifndef ___TEST__HPP___
