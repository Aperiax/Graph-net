"""

Module with Graph data structure implementation along with config manipulation
methods

"""
import numpy as np
import pandas as pd
import random 


class Graph: 
    """
    my implementation of some rudimentary graph DS
    to easily represent the cities and distances
    """

    # NOTE: so far budu pracovat s integer reprezentacema, aby se mi s tím dobře dělalo
    def __init__(self, a = 5, density = 1): 
        self.number_of_vertices = a
        self.shape = (a, a)
        self.adjacency_matrix = np.zeros(self.shape, dtype=int) 
        self._size = a 
        self._vertex_list = np.array([""]*self._size) 
        self.density = density
        self.edge_list = list()
    
    def __len__(self): 
        return  self.number_of_vertices
    
    def __str__(self):
        labels = [chr(65 + i) for i in range(self.size)]
        df = pd.DataFrame(self.get_adjacency_matrix, index=labels, columns=labels)
        return f"{df}"

    def connections_number(self): 
        max_connections_possible = (self._size * (self._size - 1)) / 2 
        connections_num = int(self.density*max_connections_possible)
        return connections_num

    @property
    def size(self):
        return self._size 

    @size.setter
    def size(self, newsize:int): 
        self._size = newsize


    @property 
    def vertex_list(self):
        vertex_list = self.vertex_list
        return vertex_list
    
    @vertex_list.setter
    def vertex_list(self, number_of_vertices):
        newlist = np.array([""]*self._size) 
        self._vertex_list = newlist

    @property 
    def get_adjacency_matrix(self): 
        matrix = self.adjacency_matrix
        return matrix
    

    def make_an_edge(self, loc: tuple): 
        """
        updates the adjacency matrix with a new edge 
        """ 
        if loc[0] == loc[1]: 
            raise ValueError("Location indices cannot be equal to each other")        

        try:
            if 0 <= loc[0] <= self.size and 0 <= loc[1] <= self.size: 
                self.adjacency_matrix[loc[0]][loc[1]] = self.adjacency_matrix[loc[1]][loc[0]] = 1
        except: 
            raise IndexError("Location indices out of bounds")


    def add_a_vertex(self, index:int, vertex_name:str): 
        """
        Adds a new vertex name to the list
        """        
        if 0 <= index <= self.size:
            self.vertex_list[index] = vertex_name
        else: 
            raise ValueError("Invalid parameters given")        


    def weight_an_edge(self, edge:tuple) -> np.ndarray:
        """
        adds a weight score onto a preexisting the edge 

        #Parameters
        :a,b: int
        """
        a, b = edge
        weight = np.random.randint(5,60)
        if self.adjacency_matrix[a][b] == 1: 
            self.adjacency_matrix[a][b] = self.adjacency_matrix[b][a] = weight
        else: 
            raise ValueError(f"No edge on ({a}, {b})")


    def check_symmetric(self) -> bool:
        """
        A debugging method to check if the generated matrices are symmetric
        """
        return np.allclose(self.adjacency_matrix, self.adjacency_matrix.T)
    
    
    def build_new_config(self):

        """ 
        Creates new config (graph) adjacency matrix based on the desired graph density,
        which is set to be random by default, but can be specified.

        # PARAMETERS
        :none
        # Returns 
        Adjacency matrix depicting skeleton of the graph
                          
                     index: | size: 
        [[0 2 0 0]     0    |   1
        [2 0 0 0]           |
        [0 0 0 0]           |
        [0 0 0 0]]    a-1   |   4
        """
        possible_edges = [(i,j) for i in range(self.size) for j in range(i+1, self.size)]
        num_edges = self.connections_number()

        selected_edges = random.sample(possible_edges, num_edges)
        # generate the edge list and save it in a class parameter to be used later
        self.edge_list = selected_edges
        for edge in selected_edges: 
            self.make_an_edge(edge)
            self.weight_an_edge(edge)
            print(f"Connection added: {edge}")
        