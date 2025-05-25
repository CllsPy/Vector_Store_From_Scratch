import numpy as np

class VectorStore:
    def __init__(self):
        self.vector_data = {} # store vectors here
        self.vector_index = {} # summary of elements
    
    def add_vector(self, vector_id, vector):
        """
        add a vector to the vector store

        Args:
            vector_id (str or int): id for the vector 
            vector (numpy.darray): the vector itself
        """
        self.vector_data[vector_id] = vector
        self._update_index(vector_id, vector)

    def get_vector(self, vector_id):
        """
        get the vector

        Args:
            vector_id(str or int): unique id
        Returns:
            numpy.darray or None (if found/If not found)
        """

        return self.vector_data[vector_id]
    
    def _update_index(self, vector_id, vector):
        """
        update index when add a new vector

        Arguments:
            vector_id (str, int): unique id
            vector (np.darray): vector
        """

        for existing_id, existing_vector in self.vector_data.items():
            similarity = np.dot(vector, existing_vector)  / (np.linalg.norm(vector) * np.linalg.norm(existing_vector))
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][vector_id] = similarity
        
    
    def find_similar_vectors(self, query_vector, num_results=5):
        """
        fodun similar vector acording a query

        Args:
            query_vector(numpy.ndarray): the vector that will be used
            num_results(int): number of similar vectors
        
        Returns:
            list: A list of (vector_id, similarity_score) yuples for the 5 similar vectors
        """
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity))

            results.sort(key=lambda x: x[1], reverse=True)

            return results[:num_results]
        



    

    


    


    

    
