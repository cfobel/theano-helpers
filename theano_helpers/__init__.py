import pprint
import cStringIO as StringIO

import theano
import theano.tensor as T
import nested_structures


def what_am_i(node):
    if node.owner:
        node, args = extract_op(node)
    if isinstance(node, theano.scalar.basic.Mul):
        return 'multiply'
    elif isinstance(node, theano.scalar.basic.IntDiv):
        return 'divide_floor'
    elif isinstance(node, theano.scalar.basic.TrueDiv):
        return 'divide'
    elif isinstance(node, theano.scalar.basic.Add):
        return 'add'
    elif isinstance(node, theano.scalar.basic.Sub):
        return 'subtract'
    elif isinstance(node, theano.tensor.elemwise.DimShuffle):
        return 'broadcast'
    elif isinstance(node, theano.scalar.basic.Pow):
        return 'pow'
    elif isinstance(node, theano.scalar.basic.Sqr):
        return 'sqr'
    elif isinstance(node, theano.scalar.basic.Sqrt):
        return 'sqrt'
    elif isinstance(node, theano.tensor.TensorConstant):
        return node.value
    elif isinstance(node, theano.tensor.TensorVariable):
        return node.name


operation = lambda v: v.owner.op.scalar_op if hasattr(v.owner.op, 'scalar_op') else v.owner.op
arguments = lambda v: v.owner.inputs
extract_op = lambda t: (operation(t), arguments(t)) if t.owner else t
extract_node = lambda t: (t, arguments(t)) if t.owner else t

# TODO:
#
#  - Apply `extract_op` recursively to construct nested operation/argument
#    tree.
#   * Perhaps we can get this in a form that will work with the
#     `nested_structures` Python package.
#  - Generate code for `thrust::transform_iterator` to implement element-wise
#    operator, using names of leaf nodes as `DeviceVectorView` variable names.
#  - Does Theano provide any notation for indirect vector element access,
#    equivalent to `permutation_iterator`?
#   * Yes! See [`TensorVariable.take`][1].
#
# [1]: http://deeplearning.net/software/theano/library/tensor/basic.html#tensor._tensor_py_operators.take
def extract(node):
    try:
        #op, args = extract_op(node)
        op, args = extract_node(node)
        if args is None:
            return op
        else:
            return (op, map(extract, args))
    except (ValueError, TypeError):
        # `node` is a leaf node, *not* an operation.
        return node


class DataFlowGraph(object):
    def __init__(self, operation):
        self.tree = nested_structures.apply_depth_first([extract(operation)],
                                                        lambda node, *args:
                                                        node, as_dict=True)

    def collect(self, func=None):
        if func is None:
            func = lambda key, *args: key
        return nested_structures.dict_collect(self.tree, func)

    def flatten(self):
        return self.collect()

    def __str__(self):
        return pprint.pformat(map(what_am_i, self.flatten()))


class Expression(object):
    def __init__(self):
        self.output = StringIO.StringIO()

    def pre(self, v, node, parents, first, last):
        self.output.write('(%s)%s' % (v.type.dtype, what_am_i(v)))
        if node.children is not None:
            self.output.write('(')

    def post(self, v, node, parents, first, last):
        if node.children is not None:
            self.output.write(')')
        if not last:
            self.output.write(', ')


if __name__ == '__main__':
    # Example usage
    x = T.ivector('x')
    y = T.ivector('y')

    dfg = DataFlowGraph(2 * x)
