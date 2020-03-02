import numpy as np
from tqdm import tqdm
import scipy.spatial.distance  # type: ignore


def k_center_greedy(X, s, b):
    '''
    Args
    - X: np.array, shape [n, d]
    - s: list of int, indices of X that have already been selected
    - b: int, new selection budget

    Returns: np.array, shape [b], type int64, newly selected indices
    '''
    n = X.shape[0]
    p = np.setdiff1d(np.arange(n), s, assume_unique=True)  # pool indices
    sel = np.empty(b, dtype=np.int64)

    sl = len(s)
    D = np.zeros([sl + b, len(p)], dtype=np.float32)
    D[:sl] = scipy.spatial.distance.cdist(X[s], X[p], metric='euclidean')  # shape (|s|,|p|)
    mins = np.min(D[:sl], axis=0)  # vector of length |p|
    cols = np.ones(len(p), dtype=bool)  # columns still in use

    for i in tqdm(range(b), desc="Greedy k-Centers"):
        j = np.argmax(mins)
        u = p[j]
        sel[i] = u

        if i == b - 1:
            break

        mins[j] = -1
        cols[j] = False

        # compute dist between selected point and remaining pool points
        r = sl + i + 1
        D[r, cols] = scipy.spatial.distance.cdist(X[u:u+1], X[p[cols]])[0]
        mins = np.minimum(mins, D[r])

    return sel


def k_center_greedy_slow(X, s, b):
    '''
    Args
    - X: np.array, shape [n, d]
    - s: list of int, indices of X representing the existing pool
    - b: int, selection budget

    Returns: np.array, shape [b], type int64, newly selected indices
    '''
    n = X.shape[0]
    p = np.setdiff1d(np.arange(n), s, assume_unique=True).tolist()  # pool indices
    sel = list(s)

    for i in range(b):
        D = scipy.spatial.distance.cdist(X[sel], X[p], metric='euclidean')  # shape (|s|,|p|)
        j = np.argmax(np.min(D, axis=0))
        u = p[j]
        sel.append(u)
        p.pop(j)

    return np.asarray(sel[-b:])


if __name__ == '__main__':
    import time
    for i in range(10):
        n, d = np.random.randint(10, 1000, size=2)
        X = np.random.randn(n, d)
        s0_size = np.random.randint(1, int(n/2))
        s = np.random.choice(n, size=s0_size)
        b = np.random.randint(1, int((n - s0_size) / 2))
        start = time.time()
        fast = k_center_greedy(X, s, b)
        fast_time = time.time() - start
        start = time.time()
        slow = k_center_greedy_slow(X, s, b)
        slow_time = time.time() - start
        assert np.all(fast == slow)
        print(f'{i}: (n={n}, d={d}, b={b}), fast {fast_time:.2f}, slow {slow_time:.2f}')
