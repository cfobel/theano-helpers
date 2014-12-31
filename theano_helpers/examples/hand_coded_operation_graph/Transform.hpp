#ifndef ___TRANSFORM__HPP___
#define ___TRANSFORM__HPP___

#include <stdint.h>
#include <thrust/functional.h>
#include <unpack_args.hpp>
#include <thrust/iterator/transform_iterator.h>

typedef float float32_t;
typedef double float64_t;

namespace theano_helpers {
  template <typename BinaryOp, typename A, typename B>
  struct Transform2 {
    typedef typename BinaryOp::result_type result_type;
    typedef A A_iterator;
    typedef B B_iterator;
    typedef thrust::tuple<A_iterator, B_iterator> args_tuple;
    typedef thrust::zip_iterator<args_tuple> args_iterator;
    typedef unpack_binary_args<BinaryOp> unpacked_op;
    typedef thrust::transform_iterator<unpacked_op, args_iterator> default_iterator;
    typedef thrust::transform_iterator<thrust::identity<result_type>, default_iterator> iterator;

    iterator operator() (A const &a, B const &b, BinaryOp op) {
      /* As the name implies, `BinaryOp` accepts two arguments.  However, since
       * we are creating a `transform_iterator` around `zip_iterator`, we must
       * wrap the `BinaryOp` with an `unpack_binary_args` adapter to unpack the
       * arguments from the zipped tuple. */
      unpacked_op u_op(op);

      return iterator(
        default_iterator(
          thrust::make_zip_iterator(thrust::make_tuple(a, b)), u_op),
        thrust::identity<result_type>());
    }
  };

}  // end namespace

#endif  // #ifndef ___TRANSFORM__HPP___
