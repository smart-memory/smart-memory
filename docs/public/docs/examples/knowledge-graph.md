# Knowledge Graph Example

This example demonstrates how to build a dynamic knowledge graph using SmartMemory that automatically discovers relationships between concepts and visualizes knowledge networks.

## Overview

The Knowledge Graph example showcases SmartMemory's ability to:

- **Automatic Relationship Discovery**: Find connections between concepts
- **Graph Visualization**: Display knowledge networks
- **Semantic Clustering**: Group related concepts
- **Interactive Exploration**: Navigate through knowledge relationships
- **Zettelkasten Integration**: Build interconnected note networks with automatic linking

## Implementation

### Basic Knowledge Graph

```python
from smartmemory import SmartMemory
import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeGraph:
    def __init__(self):
        self.memory = SmartMemory(
            config={
                "graph": {"backend": "FalkorDBBackend"},
                "evolution": {"relationship_discovery": {"enabled": True}}
            }
        )
        self.graph = nx.Graph()
    
    def add_knowledge(self, concept, description, related_concepts=None):
        """Add a concept to the knowledge graph"""
        # Store in semantic memory
        memory = self.memory.add(
            content=f"{concept}: {description}",
            memory_type="semantic",
            metadata={
                "concept": concept,
                "description": description,
                "related_concepts": related_concepts or []
            }
        )
        
        # Add to NetworkX graph
        self.graph.add_node(concept, description=description, memory_id=memory.id)
        
        # Add explicit relationships
        if related_concepts:
            for related in related_concepts:
                self.graph.add_edge(concept, related, relationship="related")
        
        return memory
    
    def discover_relationships(self, similarity_threshold=0.7):
        """Discover implicit relationships between concepts"""
        concepts = list(self.graph.nodes())
        
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Get memories for both concepts
                memory1 = self.memory.search(concept1, memory_type="semantic", max_results=1)[0]
                memory2 = self.memory.search(concept2, memory_type="semantic", max_results=1)[0]
                
                # Calculate similarity
                similarity = self.memory.calculate_similarity(memory1, memory2)
                
                if similarity > similarity_threshold:
                    self.graph.add_edge(
                        concept1, concept2, 
                        relationship="similar",
                        similarity=similarity
                    )
```

### Graph Visualization

```python
    def visualize_graph(self, layout="spring", figsize=(12, 8)):
        """Visualize the knowledge graph"""
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.random_layout(self.graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color='lightblue',
            node_size=1000,
            alpha=0.7
        )
        
        # Draw edges with different colors for different relationships
        edge_colors = []
        for u, v, data in self.graph.edges(data=True):
            if data.get('relationship') == 'similar':
                edge_colors.append('red')
            else:
                edge_colors.append('blue')
        
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color=edge_colors,
            alpha=0.5
        )
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        
        plt.title("Knowledge Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def get_concept_neighbors(self, concept, max_depth=2):
        """Get neighboring concepts up to a certain depth"""
        if concept not in self.graph:
            return []
        
        neighbors = []
        visited = set()
        queue = [(concept, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            
            visited.add(current)
            if depth > 0:  # Don't include the starting concept
                neighbors.append({
                    "concept": current,
                    "depth": depth,
                    "description": self.graph.nodes[current].get("description", "")
                })
            
            # Add neighbors to queue
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return neighbors
```

### Usage Example

```python
# Initialize knowledge graph
kg = KnowledgeGraph()

# Add machine learning concepts
kg.add_knowledge(
    "Neural Networks",
    "Computing systems inspired by biological neural networks",
    related_concepts=["Deep Learning", "Artificial Intelligence"]
)

kg.add_knowledge(
    "Deep Learning",
    "Machine learning using neural networks with multiple layers",
    related_concepts=["Neural Networks", "Machine Learning"]
)

kg.add_knowledge(
    "Machine Learning",
    "Algorithms that improve through experience",
    related_concepts=["Artificial Intelligence", "Data Science"]
)

kg.add_knowledge(
    "Natural Language Processing",
    "AI field focused on interaction between computers and human language",
    related_concepts=["Machine Learning", "Linguistics"]
)

# Discover additional relationships
kg.discover_relationships(similarity_threshold=0.6)

# Visualize the graph
kg.visualize_graph()

# Explore concept relationships
neighbors = kg.get_concept_neighbors("Machine Learning", max_depth=2)
for neighbor in neighbors:
    print(f"Depth {neighbor['depth']}: {neighbor['concept']}")
```

### Advanced Features

```python
    def get_concept_clusters(self, algorithm="louvain"):
        """Identify clusters of related concepts"""
        if algorithm == "louvain":
            import community
            clusters = community.best_partition(self.graph)
        else:
            # Use simple connected stages
            clusters = {}
            for i, component in enumerate(nx.connected_components(self.graph)):
                for node in component:
                    clusters[node] = i
        
        return clusters
    
    def find_knowledge_paths(self, start_concept, end_concept):
        """Find paths between two concepts"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph, start_concept, end_concept, cutoff=4
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_central_concepts(self, metric="betweenness"):
        """Find the most central/important concepts"""
        if metric == "betweenness":
            centrality = nx.betweenness_centrality(self.graph)
        elif metric == "degree":
            centrality = nx.degree_centrality(self.graph)
        else:
            centrality = nx.closeness_centrality(self.graph)
        
        return sorted(centrality.items(), key=lambda x: x[1], reverse=True)
```

## Features

- **Automatic Relationship Discovery**: Uses similarity metrics to find connections
- **Multiple Visualization Layouts**: Spring, circular, and random layouts
- **Concept Clustering**: Group related concepts automatically
- **Path Finding**: Discover knowledge paths between concepts
- **Centrality Analysis**: Identify key concepts in the network
- **Interactive Exploration**: Navigate through concept relationships

This example demonstrates how SmartMemory can automatically build and maintain knowledge graphs that evolve as new information is added, providing powerful tools for knowledge exploration and discovery.
