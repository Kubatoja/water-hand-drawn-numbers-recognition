import numpy as np

class Node():
    def __init__(self, ref, vecs, labels):
        self._ref = ref
        self._vecs = vecs
        self._left = None
        self._right = None
        self._labels = labels

    @property
    def labels(self):
        return self._labels

    @property
    def ref(self):
        return self._ref

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def vectors(self):
        return self._vecs

    def split(self, k, imb):
        #k - noumber of vectors in leaf
        #imb - max imbalance
        if len(self._vecs) <= k:
            return  False

        for n in range(5):
            left_vecs = []
            right_vecs = []

            left_labels = []
            right_labels = []

            # take two random indexes and set as left and right halves
            left_ref = self._vecs.pop(np.random.randint(len(self._vecs)))
            right_ref = self._vecs.pop(np.random.randint(len(self._vecs)))

            # split vectors into halves
            for i, vec in enumerate( self._vecs):
                dist_l = np.linalg.norm(vec - left_ref)
                dist_r = np.linalg.norm(vec - right_ref)
                if dist_l < dist_r:
                    left_vecs.append(vec)
                    left_labels.append(self._labels[i])
                else:
                    right_vecs.append(vec)
                    right_labels.append(self._labels[i])

            # check to make sure that the tree is mostly balanced
            r = len(left_vecs) / len(self._vecs)
            if r < imb and r > (1 - imb):
                self._left = Node(left_ref, left_vecs, left_labels)
                self._right = Node(right_ref, right_vecs, right_labels)
                return True

            # redo tree build process if imbalance is high
            self._vecs.append(left_ref)
            self._vecs.append(right_ref)

        return False
def _select_nearby(node, q, thresh = 0):

    if not node.left or not node.right:
        return ()
    dist_l = np.linalg.norm(q - node.left.ref)
    dist_r = np.linalg.norm(q - node.right.ref)
    if np.abs(dist_l - dist_r) < thresh:
        return (node.left, node.right)
    if dist_l < dist_r:
        return (node.left,)
    return (node.right,)


def _build_tree(node, K , imb):
    """Recurses on left and right halves to build a tree.
    """
    node.split(K=K, imb=imb)
    if node.left:
        _build_tree(node.left, K=K, imb=imb)
    if node.right:
        _build_tree(node.right, K=K, imb=imb)


def build_forest(vecs, labels, N: int = 32, K: int = 64, imb: float = 0.95):
    """Builds a forest of `N` trees.
    """
    forest = []
    for _ in range(N):
        root = Node(None, vecs, labels)
        _build_tree(root, K, imb)
        forest.append(root)
    return forest


def _query_linear(vecs, q: np.ndarray, k: int):
    return sorted(vecs, key=lambda v: np.linalg.norm(q-v))[:k]


def _query_tree(root: Node, q: np.ndarray, k: int):
    """Queries a single tree.
    """

    pq = [root]
    nns = []
    while pq:
        node = pq.pop(0)
        nearby = _select_nearby(node, q, thresh=0.05)

        # if `_select_nearby` does not return either node, then we are at a leaf
        if nearby:
            pq.extend(nearby)
        else:
            nns.extend(node.vecs)

    # brute-force search the nearest neighbors
    return _query_linear(nns, q, k)


def query_forest(forest, q, k: int = 10):
    nns = set()
    for root in forest:
        # merge `nns` with query result
        res = _query_tree(root, q, k)
        nns.update(res)
    return _query_linear(nns, q, k)