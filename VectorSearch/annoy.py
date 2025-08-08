from typing import List

import numpy as np

from Tester.otherModels import VectorNumberData


class Ann:
    def __init__(self, vectors: List[VectorNumberData], N=32, K=64, imb=0.95):
        vecs = [v.vector for v in vectors]
        labels = [v.label for v in vectors]

        self.forest = build_forest(np.array(vecs), np.array(labels), N, K, imb)


    def predict_label(self, q, k=10):
        return approximate_label(self.forest, np.array(q), k)


class Node:
    def __init__(self, ref, vecs, labels):
        self._ref = ref
        self._vecs = np.array(vecs, dtype=np.float32)  # Przechowuj jako float32 dla efektywności
        self._labels = np.array(labels, dtype=np.int64)
        self._left = None
        self._right = None

    @property
    def is_leaf(self):
        return self._left is None and self._right is None

    # Propercje pozostają bez zmian (ref, vecs, labels, left, right)

    def split(self, k, imb):
        if len(self._vecs) <= k:
            return False

        original_vecs = self._vecs
        original_labels = self._labels

        for _ in range(5):
            if len(original_vecs) < 2:
                return False

            # Losowanie dwóch różnych indeksów
            idx1, idx2 = np.random.choice(len(original_vecs), np.random.choice(len(original_vecs)))
            while idx1 == idx2:
                idx2 = np.random.choice(len(original_vecs))

            left_ref = original_vecs[idx1]
            right_ref = original_vecs[idx2]

            # Obliczanie odległości do referencji
            diff_left = original_vecs - left_ref
            diff_right = original_vecs - right_ref
            dists_left = np.einsum('ij,ij->i', diff_left, diff_left)  # Kwadrat odległości
            dists_right = np.einsum('ij,ij->i', diff_right, diff_right)

            # Podział wektorów
            mask = dists_left < dists_right
            left_mask = np.full(len(original_vecs), False)
            left_mask[idx1] = True  # Zawsze dodajemy referencję
            left_mask |= mask

            # Sprawdzenie balansu
            left_count = np.count_nonzero(left_mask)
            right_count = len(original_vecs) - left_count
            ratio = left_count / (left_count + right_count)

            if imb < ratio < (1 - imb):
                # Tworzenie dzieci
                self._left = Node(left_ref, original_vecs[left_mask], original_labels[left_mask])
                self._right = Node(right_ref, original_vecs[~left_mask], original_labels[~left_mask])

                # Oznacz jako wewnętrzny węzeł
                self._vecs = None
                self._labels = None
                return True

        return False


# Optymalizacja funkcji pomocniczych
def _select_nearby(node, q, thresh=0):
    if node.is_leaf:
        return ()

    dist_l = np.linalg.norm(q - node.left.ref)
    dist_r = np.linalg.norm(q - node.right.ref)

    if abs(dist_l - dist_r) < thresh:
        return (node.left, node.right)
    return (node.left,) if dist_l < dist_r else (node.right,)


def _build_tree(node, K, imb):
    if node.split(K, imb):
        _build_tree(node.left, K, imb)
        _build_tree(node.right, K, imb)


def build_forest(vecs, labels, N=32, K=64, imb=0.95):
    return [Node(None, vecs, labels) for _ in range(N)]


def _query_linear(vecs, labels, q, k):
    if len(vecs) == 0:
        return np.empty((0, q.shape[0])), np.empty(0)

    dists = np.linalg.norm(vecs - q, axis=1)
    idx = np.argpartition(dists, k)[:k]
    return vecs[idx], labels[idx]  # Zwracamy odpowiednie etykiety


def query_forest(forest, q, k=10):
    candidates = []
    candidate_labels = []  # Nowa lista na etykiety

    for tree in forest:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if node.is_leaf:
                candidates.extend(node._vecs)
                candidate_labels.extend(node._labels)  # Zbieramy etykiety
            else:
                nodes.extend(_select_nearby(node, q))

    if not candidates:
        return np.empty((0, q.shape[0])), np.empty(0)

    # Przekazujemy obie listy do _query_linear
    return _query_linear(np.array(candidates), np.array(candidate_labels), q, k)

def approximate_label(forest, q, k=10):
    _, labels = query_forest(forest, q, k)
    if labels.size == 0:
        return None
    return np.bincount(labels).argmax() if len(labels) > 0 else None