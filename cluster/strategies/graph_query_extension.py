
    def get_available_transitions(self, state_vector: List[float], architecture: str) -> List[Dict]:
        """
        Returns all known transitions (actions) from the anchor nearest to the current state.
        
        Returns:
            List[Dict]: List of edge data dictionaries containing:
                - action (tool name)
                - impact_vector
                - visits
                - avg_reward (for legacy compatibility)
        """
        if not self.graph:
            return []
            
        # 1. Find Anchor
        # We reuse the internal logic but don't spawn new nodes here, just find existing.
        # Simple KNN with k=1.
        vec = np.array(state_vector).flatten()
        best_node = None
        min_dist = float('inf')
        
        for node, data in self.graph.nodes(data=True):
             # Strict Architecture Match
             if data.get("architecture") != architecture:
                 continue
                 
             stored_vec = np.array(data["vector"])
             dist = np.linalg.norm(vec - stored_vec)
             
             if dist < min_dist:
                 min_dist = dist
                 best_node = node
        
        # If no node found or too far, return empty (Explorer mode will handle this)
        if not best_node or min_dist > SIMILARITY_THRESHOLD * 2.0: # Relased threshold for querying
            return []
            
        # 2. Get Outgoing Edges
        transitions = []
        if self.graph.has_node(best_node):
            for u, v, key, data in self.graph.edges(best_node, data=True, keys=True):
                # Standardize return format
                transitions.append({
                    "action": key,
                    "target_node": v,
                    "impact_vector": data.get("impact_vector", []),
                    "visits": data.get("visits", 0),
                    "raw_rewards": data.get("raw_rewards", [])
                })
                
        return transitions

    def get_node_data(self, node_id: str) -> Dict:
        """Returns data for a specific node (vector, visits)."""
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        return {}
