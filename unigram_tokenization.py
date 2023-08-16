import pandas as pd
import copy
import numpy as np
from node import Node
from edge import Edge
from graph import Graph
from helpful_functions import read_file
from bpe_tokenization import BPETokenizer
pd.options.mode.chained_assignment = None

class UnigramTokenizer:
    def __init__(self, vocab_size=20,
                 path_to_txt="txt/sample.txt",
                 max_iterations=100):
        self.path_to_txt = path_to_txt
        self.seed_generation_max_iterations = max_iterations
        self.tokens_df = pd.DataFrame()
        self.token_frequency = {}
        self.token_probability = {}
        self.pretokenized_words_split_dict = {}
        self.words_counter_dict = {}
        self.subword_probability = {}
        self.best_path_to_subw = {}
        self.pretokenized_words_without_end = []
        self.pretokenized_words = []
        self.oov_probability = 100
        self.graph = None
        self.debug = False

    def generate_seed_from_bpe(self):
        """
        Generation of subwords seed id needed to start
        unigram algorithm. We use here the tokens generated
        in the first n iterations of bpe algorithm
        """
        bpe_tkn = BPETokenizer(path_to_txt=self.path_to_txt)
        self.tokens, self.pretokenized_words, \
            self.pretokenized_words_split_dict, self.words_counter_dict,\
            self.pretokenized_words_without_end = bpe_tkn.create_and_split_corpus()
        """
        Base tokens should always remain in the token set,
        to make sure we can tokenize every new word
        """
        self.base_tokens = self.tokens["Token"].to_list()
        self.token_seed, _ = bpe_tkn.train(self.seed_generation_max_iterations)
        self.token_seed = self.token_seed.sort_values("Frequency", ascending=False)["Token"].to_list()
        self.tokens = self.token_seed + self.base_tokens

    def tokenization_probability(self):
        """
        Tokens frequencies indicate how many times a given token was
        used in the original text. Frequency of '</w> is calculated
        separately
        """
        for token in self.tokens:
            if token not in self.token_frequency.keys():
                self.token_frequency[token] = 0
                self.token_probability[token] = 0
            if '</w>' not in token:
                for word in self.pretokenized_words_without_end:
                    self.token_frequency[token] += word.count(token)
            else:
                for word in self.pretokenized_words:
                    self.token_frequency[token] += word.count(token)
        frequencies_sum = sum(self.token_frequency.values())
        """
        Counting token choice probability based on its frequency
        """
        for token in self.tokens:
            self.token_probability[token] = -np.log(self.token_frequency[token]/frequencies_sum)

    def viterbi(self, word):
        # Create graph nodes
        graph = Graph()
        nodes = []
        self.nodes_dict = {}
        self.edges_dict = {}
        for j in range(1, len(word)+1):
            graph.add_node(Node(word[:j]))
            self.nodes_dict[word[:j]] = Node(word[:j])
        graph.add_node(Node("init"))
        self.nodes_dict["init"] = Node("init")

        nodes_ids = graph.get_nodes_ids()
        # Create graph edges
        for node in graph.get_nodes():
            node_id = node.get_node_id()
            for tkn, prob in self.token_probability.items():
                # If token starts at the beginning of the word
                if tkn == node_id:
                    graph.add_edges(Edge(Node("init"), Node(tkn), prob))
                    self.edges_dict[f'{"init"}{"-"}{tkn}'] = Edge(Node("init"), Node(tkn), prob)
                    if self.debug:
                        print(f'{"Node: "}{"init"}{" Token: "}{tkn}{" Combined: "}{tkn}')
                else:
                    if node_id+tkn in nodes_ids:
                        graph.add_edges(Edge(Node(node_id), Node(node_id+tkn), prob))
                        self.edges_dict[f'{node_id}{"-"}{node_id+tkn}'] = \
                            Edge(Node(node_id), Node(node_id+tkn), prob)
                        if self.debug:
                            print(f'{"Node: "}{node_id}{" Token: "}{tkn}{" Combined: "}{node_id + tkn}')
        if self.debug:
            print(f'{"NODES: "}{graph.get_nodes_ids()}')
            print(f'{"EDGES: "}{graph.get_edges_ids()}')
        self.graph = graph
        acceptable_paths_tokens = self.dfs(self.nodes_dict["init"], self.nodes_dict[word])
        best_path = []
        best_path_rate = np.inf
        for path in acceptable_paths_tokens:
            path_rate = 0
            for token in path:
                path_rate += self.token_probability[token]
            if path_rate < best_path_rate:
                best_path_rate = path_rate
                best_path = path
        print(f'{"Best path: "}{best_path}{" Cost: "}{best_path_rate}')

    def dfs(self, init, src):
        edges = self.graph.get_edges()
        edges_ids = self.graph.get_edges_ids()
        nodes = self.graph.get_nodes()
        nodes_ids = self.graph.get_nodes_ids()
        paths = []
        acceptable_paths = []
        acceptable_paths_ids = []
        acceptable_paths_edges_ids = []
        acceptable_paths_tokens = []
        # Initializing dfs stack with initial node inside
        stack = [init]
        unvisited_neighbors = {}
        while stack:
            if self.debug:
                print("STACK:")
                for node in stack:
                    print(node.get_node_id())
            current = stack[-1]
            # If current node has not been visited yet
            if current not in unvisited_neighbors.keys():
                current_neighbors = self.graph.get_neighbors(current)
                unvisited_neighbors[current] = current_neighbors
            if not unvisited_neighbors[current]:
                paths.append(copy.deepcopy(stack))
                # Updating the list of unvisited neighbors corresponding
                # to the node that is popped of the stack
                unvisited_neighbors[stack[-1]] = \
                    self.graph.get_neighbors(stack[-1])
                stack.pop()
            else:
                stack.append(unvisited_neighbors[current][0])
                unvisited_neighbors[current].pop(0)
        for path in paths:
            if path[-1].get_node_id() == src.get_node_id():
                acceptable_paths.append(path)

        for path in acceptable_paths:
            acceptable_path_ids = []
            for node in path:
                acceptable_path_ids.append(node.get_node_id())
            acceptable_paths_ids.append(acceptable_path_ids)

        for acceptable_path_ids in acceptable_paths_ids:
            acceptable_paths_edge_ids = []
            acceptable_paths_token = []
            for i in range(len(acceptable_path_ids)-1):
                acceptable_paths_edge_ids.append(
                    self.edges_dict[f'{acceptable_path_ids[i]}{"-"}{acceptable_path_ids[i+1]}'])
                if acceptable_path_ids[i] == "init":
                    acceptable_paths_token.append(acceptable_path_ids[i+1])
                else:
                    acceptable_paths_token.append(
                        acceptable_path_ids[i+1][len(acceptable_path_ids[i]):])
            acceptable_paths_edges_ids.append(acceptable_paths_edge_ids)
            acceptable_paths_tokens.append(acceptable_paths_token)
        return acceptable_paths_tokens

    def train(self):
        for word in self.pretokenized_words:
            self.viterbi(word)

tkn = UnigramTokenizer()
tkn.generate_seed_from_bpe()
tkn.tokenization_probability()
tkn.train()
