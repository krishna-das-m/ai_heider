import numpy as np

class Edge2Vec:
    def __init__(self, graph, embedding_dim=128, walk_length=100, num_walks=10, p:float=1, q:float=1,
                weight_key='weight'):
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.walks = self._simulate_walks()
    
    def get_next_node(self, current, previous):
        G = self.graph
        alphas = []
        neighbors = list(G.neighbors(current))
        for neighbor in neighbors:
            weight = abs(G[current][neighbor][self.weight_key])
            if neighbor == previous:
                alpha = weight* 1/self.p
            elif G.has_edge(neighbor, previous):
                alpha = weight
            else:
                alpha = weight * 1/self.q
            alphas.append(alpha)
        probs = [alpha/sum(alphas) for alpha in alphas] #normalized edge transition probabilities
        next = np.random.choice(neighbors,1,p=probs)[0]
        return next

    def edge2vec(self, start_node):
        G = self.graph
        walk = [start_node]
        # for i in range(walk_length-1):
        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = list(G.neighbors(current)) 
            if not neighbors:
                break
            if len(walk)==1:
                # neighbors = list(G.neighbors(current))
                next = np.random.choice(neighbors)
            else:
                previous = walk[-2]
                next = self.get_next_node(G, current, previous)
            walk.append(next)
        return walk

    def _simulate_walks(self):
        walks = []
        G = self.graph
        nodes = list(G.nodes)
        for _ in range(self.num_walks):
            np.random.shuffle(nodes) # Shuffle nodes to ensure randomness
            for node in nodes:
                walk = self.edge2vec(node)
                walks.append(walk)
        return walks
