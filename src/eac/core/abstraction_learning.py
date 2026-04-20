import torch
from collections import defaultdict

class AbstractionLearning:
    def __init__(self, monitor):
        self.monitor = monitor
        self.concepts = defaultdict(dict)

    def compute_boundary(self, simplices):
        faces = sorted(list(set([f for s in simplices for f in s])))
        f2i = {f: i for i, f in enumerate(faces)}
        boundary = torch.zeros((len(faces), len(simplices)))
        for j, s in enumerate(simplices):
            for i, f in enumerate(s):
                boundary[f2i[f], j] = 1.0 if i % 2 == 0 else -1.0
        return boundary

    def process(self, observation):
        simplices = [(0,1), (1,2), (2,0)]
        b1 = self.compute_boundary(simplices)
        rank = torch.linalg.matrix_rank(b1)
        betti1 = len(simplices) - rank.item()
        return {"betti1": betti1}
