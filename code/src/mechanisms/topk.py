import numpy as np

def topk_sparsify(update, k):
    """
    Sparsify a flat parameter update by keeping only the top-k
    values.

    All values except the k largest (by absolute value) are set
    to zero.
    This should reduce communication cost by only transmitting
    the most significant gradient components.

    Args:
        update (np.ndarray): flat parameter update vector.
        k (float):           fraction of values to keep, e.g. 0.01 for 1%.

    Returns:
        np.ndarray: sparsified update with only top-k values non-zero.
    """
    num_keep = max(1, int(len(update) * k))

    # Find indices of k largest values by absolute value
    top_indices = np.argsort(np.abs(update))[-num_keep:]

    # Zero everything else out
    sparsified = np.zeros_like(update)
    sparsified[top_indices] = update[top_indices]

    return sparsified


INDEX_DTYPE_BYTES = 4  # 32-bit index -- enough to address any of this project's model sizes


def update_size_bytes(updated_parameters, use_topk, sparsified_update=None):
    """
    Bytes needed to transmit one client's update this round.

    use_topk=False: this matches the actual current cost -- the dense vector
    this function sizes is genuinely what fit() returns and what Flower/Ray
    serializes across the client/server boundary today, so
    total parameter count * dtype size is real, not estimated.

    use_topk=True: this is a logical/computed number, NOT what actually
    crosses the wire in this simulation today. fit() still returns a dense,
    full-shaped parameter vector regardless of use_topk (Flower's FedAvg-style
    aggregation needs every client's arrays to line up by position), so a
    TopK config's real current bytes-on-wire equals the dense case above,
    unaffected by topk_ratio. What's computed here instead is what a real
    sparse encoding would cost if transmitted as only sparsified_update's
    non-zero entries, each as an (index, value) pair -- the receiver can't
    place a value without knowing which position it belongs to. See
    src/README.md's "TopK doesn't reduce actual bytes transmitted in the
    simulation" entry for why that's not implemented (yet).

    Args:
        updated_parameters (list[np.ndarray]): this round's dense parameter
                                                arrays, as returned by fit().
                                                Only used when use_topk=False.
        use_topk (bool):              whether TopK sparsification is enabled.
        sparsified_update (np.ndarray | None): the sparsified flat delta from
                                                topk_sparsify(). Required when
                                                use_topk=True.

    Returns:
        int: size in bytes.
    """
    if use_topk:
        nonzero = np.count_nonzero(sparsified_update)
        return int(nonzero * (sparsified_update.itemsize + INDEX_DTYPE_BYTES))
    return int(sum(p.size * p.itemsize for p in updated_parameters))


def unflatten(flat, shapes):
    """
    Split a flat vector back into a list of arrays with the given shapes.

    Inverse of concatenating a list of per-layer arrays into one flat
    vector (`np.concatenate([p.flatten() for p in parameters])`), a pattern
    used throughout this project (topk, FLTrust, the scaling/scrambling
    attacks) to operate on a whole model's parameters as one vector.

    Args:
        flat (np.ndarray):    flat vector, total size must match sum(shapes).
        shapes (list[tuple]): target shape for each output array, in order.

    Returns:
        list[np.ndarray]: arrays reshaped per `shapes`, in the same order.
    """
    arrays = []
    index = 0
    for shape in shapes:
        size = int(np.prod(shape))
        arrays.append(flat[index:index + size].reshape(shape))
        index += size
    return arrays


if __name__ == '__main__':
    update = np.array([0.01, -0.5, 0.001, 0.8, -0.3, 0.002, 0.4, -0.1])
    print(f"Original update: {update}")
    print(f"Non zero values: {np.count_nonzero(update)}")

    for k in [0.5, 0.25]:
        sparsified = topk_sparsify(update, k)
        print(f"k={k:.0%} -> kept {np.count_nonzero(sparsified)}/{len(update)} values")
        print(f"{sparsified}")

    shapes = [(2, 2), (4,)]
    flat = np.arange(8.0)
    restored = unflatten(flat, shapes)
    print(f"\nunflatten({flat}, {shapes}):")
    for arr in restored:
        print(arr)

    dense_params = [np.zeros((2, 2), dtype=np.float32), np.zeros(4, dtype=np.float32)]
    dense_bytes = update_size_bytes(dense_params, use_topk=False)
    print(f"\ndense update_size_bytes (8 float32 params, no TopK): {dense_bytes} bytes")

    sparsified = topk_sparsify(update, 0.25)  # 8 values, k=0.25 -> 2 kept
    sparse_bytes = update_size_bytes(dense_params, use_topk=True, sparsified_update=sparsified)
    print(f"TopK update_size_bytes (k=0.25 -> {np.count_nonzero(sparsified)} nonzero, "
          f"{sparsified.itemsize}B value + {INDEX_DTYPE_BYTES}B index each): {sparse_bytes} bytes")
