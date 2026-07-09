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
