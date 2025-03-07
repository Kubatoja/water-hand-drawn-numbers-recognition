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
    def vecs(self):
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

            rand_index = np.random.randint(len(self._vecs))
            left_ref = self._vecs[rand_index]
            left_label_ref = self._labels[rand_index]
            self._vecs = np.delete(self._vecs, rand_index,axis=0)
            self._labels = np.delete(self._labels, rand_index, axis=0)

            rand_index = np.random.randint(len(self._vecs))
            right_ref = self._vecs[rand_index]
            right_label_ref = self._labels[rand_index]
            self._vecs = np.delete(self._vecs, rand_index, axis=0)
            self._labels = np.delete(self._labels, rand_index, axis=0)

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
            self._vecs = np.concatenate(([left_ref], self._vecs))
            self._vecs = np.concatenate(([right_ref], self._vecs))
            self._labels = np.concatenate(([left_label_ref], self._labels))
            self._labels = np.concatenate(([right_label_ref], self._labels))

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
    node.split(k=K, imb=imb)
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


def _query_linear(vecs, labels, q: np.ndarray, k: int):
    labels = np.array(labels)
    vecs = np.array(vecs)
    sorted_indices = np.argsort([np.linalg.norm(q - v) for v in vecs])[:k]

    nearest_vecs = vecs[sorted_indices]
    nearest_labels = labels[sorted_indices]
    return nearest_vecs, nearest_labels


def _query_tree(root: Node, q: np.ndarray, k: int):
    """Queries a single tree.
    """

    pq = [root]
    nns = []
    labels = []
    while pq:
        node = pq.pop(0)
        nearby = _select_nearby(node, q, thresh=0.05)

        # if `_select_nearby` does not return either node, then we are at a leaf
        if nearby:
            pq.extend(nearby)
        else:
            nns.extend(node.vecs)
            labels.extend(node.labels)

    # brute-force search the nearest neighbors
    return _query_linear(nns, labels, q, k)


def query_forest(forest, q, k: int = 10):
    nns = set()
    labels = []
    for root in forest:
        nns_size_before_query = len(nns)
        # merge `nns` with query result
        res, label = _query_tree(root, q, k)
        nns.update(tuple(tuple(row.tolist()) for row in res))

        if len(nns) > nns_size_before_query:
            labels.extend(label.tolist())

    nns = np.array(list(nns))
    return _query_linear(nns, labels,q, k)


def approximate_label(forest, q, k: int = 10):
    query_vecs, query_labels = query_forest(forest, q, k)


    label_dict = {}


    # Iterate through the array and count occurrences of value1
    for label in query_labels:
        if label in label_dict:
            label_dict[label] += 1
        else:
            label_dict[label] = 1

    # Find the value1 with the highest count
    most_common = None
    max_count = 0
    for label, count in label_dict.items():
        if count > max_count:
            most_common = label
            max_count = count

    return most_common
