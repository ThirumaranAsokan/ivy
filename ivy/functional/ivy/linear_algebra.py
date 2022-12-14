def argmax(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    output_dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    # Decorator functions
    @to_native_arrays_and_back
    @handle_out_argument
    @handle_nestable
    @handle_exceptions
    @handle_array_like
    def decorated_function(x, axis, keepdims, output_dtype, out):
        # Original function implementation goes here
        pass

    # Call the decorated function
    return decorated_function(x, axis, keepdims, output_dtype, out)
