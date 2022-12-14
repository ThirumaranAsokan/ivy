# global
import abc
from typing import Optional, Union

# local
import ivy

class MyArray(ArrayWithActivations):
    def __init__(self, data: Union[ivy.Array, ivy.NativeArray]):
        self._data = data

    @abc.abstractmethod
    def get_data(self) -> Union[ivy.Array, ivy.NativeArray]:
        """
        Returns the data stored in the array.
        """
        pass

    @abc.abstractmethod
    def set_data(self, data: Union[ivy.Array, ivy.NativeArray]):
        """
        Sets the data stored in the array.
        """
        pass

x = MyArray(data=[-1.0, 0.0, 1.0])
y = x.relu()

    import numpy as np

    class ivyArray:
      def __init__(self, data):
        self._data = data

      def leaky_relu(self, alpha=0.2):
        return ivyArray(np.maximum(self._data, alpha * self._data))


    def leaky_relu(x, alpha=0.2):
      return ivyArray(np.maximum(x, alpha * x))


    
    def gelu(
        self: ivy.Array,
        /,
        *,
        approximate: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.gelu. This method simply wraps the
        function, and so the docstring for ivy.gelu also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([0.3, -0.1])
        >>> y = x.gelu()
        >>> print(y)
        ivy.array([ 0.185, -0.046])
        """
        return ivy.gelu(self._data, approximate=approximate, out=out)

    def sigmoid(self: ivy.Array, /, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sigmoid. This method simply wraps the
        function, and so the docstring for ivy.sigmoid also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([-1., 1., 2.])
        >>> y = x.sigmoid()
        >>> print(y)
        ivy.array([0.269, 0.731, 0.881])
        """
        return ivy.sigmoid(self._data, out=out)

    def softmax(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.softmax. This method simply wraps the
        function, and so the docstring for ivy.softmax also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input array.
        axis
            the axis or axes along which the softmax should be computed
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the softmax unit function applied element-wise.

        Examples
        --------
        >>> x = ivy.array([1.0, 0, 1.0])
        >>> y = x.softmax()
        >>> print(y)
        ivy.array([0.422, 0.155, 0.422])
        """
        return ivy.softmax(x, axis=axis, out=out)

    def softplus(
        self: ivy.Array,
        /,
        *,
        beta: Optional[Union[int, float]] = None,
        threshold: Optional[Union[int, float]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.softplus. This method simply wraps the
        function, and so the docstring for ivy.softplus also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([-0.3461, -0.6491])
        >>> y = x.softplus()
        >>> print(y)
        ivy.array([0.535,0.42])

        >>> x = ivy.array([-0.3461, -0.6491])
        >>> y = x.softplus(beta=0.5)
        >>> print(y)
        ivy.array([1.22, 1.09])

        >>> x = ivy.array([1.31, 2., 2.])
        >>> y = x.softplus(threshold=2, out=x)
        >>> print(x)
        ivy.array([1.55, 2.13, 2.13])
        """
        return ivy.softplus(self._data, beta=beta, threshold=threshold, out=out)

    def log_softmax(
        self: ivy.Array,
        /,
        *,
        axis: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.log_softmax.
        This method simply wraps the function,
        and so the docstring for ivy.log_softmax also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([-1.0, -0.98, 2.3])
        >>> y = x.log_softmax()
        >>> print(y)
        ivy.array([-3.37, -3.35, -0.0719])

        >>> x = ivy.array([2.0, 3.4, -4.2])
        >>> y = x.log_softmax(x)
        ivy.array([-1.62, -0.221, -7.82 ])
        """
        return ivy.log_softmax(self._data, axis=axis, out=out)
