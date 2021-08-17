#
# This class provides using mathematical operators.
#
import numpy as np


class Operators:

    @staticmethod
    def uniform(shape, dim=0):
        # Check that inputs are correct.
        dims = len(shape)
        if dims > 3:
            raise RuntimeError("In Operators.uniform, length of input 'shape' must must be <= 3.")
        if dim >= dims:
            raise RuntimeError("In Operators.uniform, input 'dim' must be < to length of input 'shape'.")

        # Implement the function.
        return np.full(shape, 1.0 / shape[dim])

    @staticmethod
    def one_hot(n_obs, obs):
        vec = np.zeros([n_obs])
        vec[obs] = 1
        return vec

    @staticmethod
    def expansion(x1, n, dim):
        result = np.expand_dims(x1, dim)
        return result.repeat(n, axis=dim)

    @staticmethod
    def multiplication(x1, x2, ml):
        # Create the list on non-matching dimensions
        not_ml = []
        for i in range(x1.ndim):
            if ml.count(i) == 0:
                not_ml.append(i)

        # Sequence of expansions
        x2_tmp = x2
        for i in not_ml:
            x2_tmp = Operators.expansion(x2_tmp, x1.shape[i], x2_tmp.ndim)

        # Permutation
        pl = [0] * x1.ndim
        for i in range(x1.ndim):
            try:
                pl[i] = ml.index(i)
            except ValueError:
                pl[i] = len(ml) + not_ml.index(i)
        x2_tmp = x2_tmp.transpose(pl)

        # Element-wise multiplication
        return x2_tmp * x1

    @staticmethod
    def average(x1, x2, ml, el=None):
        if el is None:
            el = []

        # Perform the element-wise multiplication
        result = Operators.multiplication(x1, x2, ml)

        # Create the reduction list, i.e. rl = ml \ el where "\" = set minus
        rl = ml
        for elem in el:
            rl.remove(elem)

        # Sort the reduction list in decreasing order
        rl.sort(reverse=True)

        # Reduction of the tensor (using a summation) along the dimension of the reduction list
        for i in rl:
            result = result.sum(i)
        return result
