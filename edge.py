class Edge:
    def __init__(self, start_node, end_node, cost):
        self.start_node = start_node
        self.end_node = end_node
        self.cost = cost
        self.edge_id = f'{start_node.get_node_id()}{"-"}{end_node.get_node_id()}'

    def __str__(self):
        return f'{"Edge from: "}{self.start_node.get_node_name()}{", to: "}' \
               f'{self.end_node.get_node_name()}{". Edge cost: "}{self.cost}'

    def get_start_node(self):
        return self.start_node

    def get_end_node(self):
        return self.end_node

    def get_edge_cost(self):
        return self.cost

    def get_edge_id(self):
        return self.edge_id