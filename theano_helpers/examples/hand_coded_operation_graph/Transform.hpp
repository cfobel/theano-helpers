#ifndef ___TRANSFORM__HPP___
#define ___TRANSFORM__HPP___

#include <thrust/functional.h>
#include <unpack_args.hpp>
#include <thrust/iterator/transform_iterator.h>


namespace theano_helpers {
  template <typename Op, typename A, typename B>
  struct Transform2 {
    typedef typename Op::result_type result_type;
    typedef A A_iterator;
    typedef B B_iterator;
    typedef thrust::tuple<A_iterator, B_iterator> args_tuple;
    typedef thrust::zip_iterator<args_tuple> args_iterator;
    typedef unpack_binary_args<Op> unpacked_op;
    typedef thrust::transform_iterator<unpacked_op, args_iterator> default_iterator;
    typedef thrust::transform_iterator<thrust::identity<result_type>, default_iterator> iterator;

    A_iterator a;
    B_iterator b;
    unpacked_op op;

    Transform2(A const &a, B const &b, Op op)
      : a(a), b(b), op(op) {}

    iterator begin() {
      return iterator(
        default_iterator(
          thrust::make_zip_iterator(thrust::make_tuple(a, b)), op),
        thrust::identity<result_type>());
    }
  };

}  // end namespace

#endif  // #ifndef ___TRANSFORM__HPP___
