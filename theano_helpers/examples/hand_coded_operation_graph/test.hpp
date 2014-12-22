#ifndef ___TEST__HPP___
#define ___TEST__HPP___

#include <thrust/iterator/iterator_traits.h>
#include <thrust/copy.h>
#include "Transform.hpp"


namespace uuid00 {
  struct Run {
    template <typename A, typename B, typename C>
    void operator() (A a, size_t N, B b, C c) {
      /* Operation:
       *
       *     c = a / b
       */
      typedef typename thrust::iterator_traits<C>::value_type result_type;
      typedef thrust::divides<result_type> Op;
      typedef theano_helpers::Transform2<Op, A, B> operator_graph_type;
      typedef typename operator_graph_type::iterator operator_graph_iterator;

      operator_graph_iterator operator_graph =
        operator_graph_type(a, b, Op()).begin();

      thrust::copy_n(operator_graph, N, c);
    }
  };

}  // end namespace

#endif  // #ifndef ___TEST__HPP___
