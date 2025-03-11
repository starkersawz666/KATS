import os
import networkx as nx


class GraphService:

    def __init__(self):
        pass

    @staticmethod
    def graph_init() -> str:
        graph_path = "data/graph/graph.graphml"
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        return graph_path

    @staticmethod
    def load_graph(graph_path: str) -> nx.Graph:
        if os.path.exists(graph_path):
            return nx.read_graphml(graph_path)
        return nx.Graph()

    @staticmethod
    def save_graph(graph, graph_path):
        nx.write_graphml(graph, graph_path)
