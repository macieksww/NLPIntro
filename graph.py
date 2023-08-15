class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.edges_ids = []
        self.nodes_ids = []

    def get_edges(self):
        return self.edges

    def get_nodes(self):
        return self.nodes

    def add_edges(self, edges):
        self.edges.append(edges)
        self.edges_ids.append(edges.get_edge_id())
        # if isinstance(edges, list):
        #     for edge in edges:
        #         if edge.get_edge_id() not in self.edges_ids:
        #             self.edges.append(edge)
        #             self.edges_ids.append(edge.get_edge_id())
        # else:
            # if edges.get_edge_id() not in self.edges_ids:
            # self.edges.append(edges)
            # self.edges_ids.append(edges.get_edge_id())

    def add_node(self, node):
        self.nodes.append(node)

    def get_edges_ids(self):
        self.edges_ids = []
        edges = self.get_edges()
        for edge in edges:
            self.edges_ids.append(edge.get_edge_id())
        return self.edges_ids

    def get_nodes_ids(self):
        self.nodes_ids = []
        nodes = self.get_nodes()
        for node in nodes:
            self.nodes_ids.append(node.get_node_id())
        return self.nodes_ids

    def get_neighbors(self, node):
        node_id = node.get_node_id()
        neighbors = []
        for edge in self.edges:
            if edge.get_start_node().get_node_id() == node_id:
                neighbors.append(edge.get_end_node())
        return neighbors


