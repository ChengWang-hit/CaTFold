"""Maximum-weight matching for CaTFold predictions."""

import networkx as nx
import torch


def decode(probabilities, threshold=0.5):
    values = probabilities.detach().cpu()
    graph = nx.Graph()
    rows, columns = torch.where(torch.triu(values, diagonal=1) > threshold)
    for row, column in zip(rows.tolist(), columns.tolist()):
        graph.add_edge(row, column, weight=float(values[row, column]))
    prediction = torch.zeros_like(probabilities)
    for row, column in nx.algorithms.matching.max_weight_matching(graph):
        prediction[row, column] = 1
        prediction[column, row] = 1
    return prediction
