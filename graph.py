import numpy as np 
import tensorflow as tf 

NODE_EMBEDDING_SIZE = 300
EDGE_EMBEDDING_SIZE = 30
HYPER_EMBEDDING_SIZE = 100 

#edges: (3, 4) = 3 <-> 4, sorted, lowest first


class Embedding(tf.keras.layers.Layer):
    def __init__(self, dimensions):
        self.dim = dimensions
        super(Embedding, self).__init__()

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.dim, input_shape=input_shape, activation=tf.nn.relu, use_bias=True)
        self.norm = tf.keras.layers.BatchNormalization()
        self._trainable_weights = self.dense.trainable_weights

    def call(self, inp, training=None):
        x = self.dense(inp)
        return self.norm(x) if training else x 

class Update(tf.keras.layers.Layer):
    def __init__(self, dimensions):
        super(Update, self).__init__()
        self.dim = dimensions

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.dim, input_shape=input_shape, use_bias=True)
        self._trainable_weights = self.dense.trainable_weights
    
    def call(self, inp):
        x = self.dense(inp)
        return tf.math.sigmoid(inp)
        
class MessagePassing(tf.keras.layers.Layer):
    def __init__(self):
        super(MessagePassing, self).__init__()
    
    def call(self, nodes, edges, adjacency):

        nodeupdates = [list() for x in range(nodes)]
        for i in adjacency: #node neighbor mixing
            nodeupdates[i[0]] += nodes[i[1]]
            nodeupdates[i[1]] += nodes[i[0]]

        for i in range(len(nodes)): #average pooling
            nodes[i] += tf.reduce_sum(nodeupdates[i])
            nodes[i] /= len(nodeupdates[i]) + 1

        for i in adjacency: #concat edge weights
            nodes[i[0]] = tf.concat(i[0], edges[i])
            nodes[i[1]] = tf.concat(i[1], edges[i])

        return nodes, edges

class GraphNN(tf.keras.Model):
    def __init__(self, input_features, edge_features, hyper_features, nemb_size, eemb_size, hemb_size, iterations, name="graphnn", **kwargs):
        super(GraphNN, self).__init__(name=name, **kwargs)
        self.nodeembedding = Embedding(nemb_size)
        self.hyperembedding = Embedding(hemb_size)

        self.msgpass = MessagePassing()
        self.update = Update(emb_size)
        self.iterations = iterations
        

    def call(self, inputs, edges, adjacency, membership):
        nodeembeddings = [] 
        edgeembeddings = {}
        
        for i in inputs:
            embeddings.append(self.embedding(i))

        for i in edges:
            edgeembeddings[i] = edges[i]#self.embedding(edges[i])

        x = None
        for _ in range(iterations):

            x = self.msgpass(nodeembeddings, edgeembeddings, adjacency)
            x = tf.map_fn(self.update, x)
        
        return x

gnn = GraphNN(120, NODE_EMBEDDING_SIZE, EDGE_EMBEDDING_SIZE, GLOBAL_EMBEDDING_SIZE, 3)

print(gnn.summary())