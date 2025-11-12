# Ontology Management

SmartMemory includes a sophisticated ontology management system that helps structure and organize knowledge for better retrieval and reasoning. This guide covers how to define, manage, and leverage ontologies in your agentic memory system.

## Overview

An ontology in SmartMemory defines:
- **Concepts and Categories**: Hierarchical knowledge structures
- **Relationships**: How concepts relate to each other
- **Properties**: Attributes and characteristics of concepts
- **Rules**: Inference and reasoning rules
- **Constraints**: Validation and consistency rules

## Core Ontology Concepts

### Concept Hierarchies

Define hierarchical knowledge structures:

```python
from smartmemory.ontology import OntologyManager, Concept, Relationship

# Initialize ontology manager
ontology = OntologyManager()

# Define concept hierarchy
technology = Concept("Technology", description="All technology-related concepts")
ai = Concept("ArtificialIntelligence", parent=technology, description="AI and ML concepts")
ml = Concept("MachineLearning", parent=ai, description="Machine learning techniques")
dl = Concept("DeepLearning", parent=ml, description="Deep learning methods")

# Add concepts to ontology
ontology.add_concepts([technology, ai, ml, dl])

# Define specific algorithms
cnn = Concept("ConvolutionalNeuralNetwork", parent=dl)
rnn = Concept("RecurrentNeuralNetwork", parent=dl)
transformer = Concept("Transformer", parent=dl)

ontology.add_concepts([cnn, rnn, transformer])
```

### Relationship Types

Define semantic relationships between concepts:

```python
# Define relationship types
ontology.define_relationship_type(
    "IS_A", 
    description="Hierarchical inheritance relationship",
    properties={"transitive": True, "reflexive": False}
)

ontology.define_relationship_type(
    "USES",
    description="One concept uses another",
    properties={"transitive": False, "symmetric": False}
)

ontology.define_relationship_type(
    "SIMILAR_TO",
    description="Concepts are similar",
    properties={"symmetric": True, "transitive": False}
)

ontology.define_relationship_type(
    "PART_OF",
    description="Component relationship",
    properties={"transitive": True, "reflexive": False}
)

# Create relationships
ontology.add_relationship(
    Relationship(cnn, "SIMILAR_TO", rnn, strength=0.7)
)

ontology.add_relationship(
    Relationship(transformer, "USES", "attention_mechanism", strength=0.9)
)
```

### Properties and Attributes

Define properties for concepts:

```python
# Define property types
ontology.define_property_type(
    "complexity",
    data_type="float",
    range=(0.0, 1.0),
    description="Complexity level of the concept"
)

ontology.define_property_type(
    "application_domain",
    data_type="list",
    allowed_values=["nlp", "computer_vision", "robotics", "speech"],
    description="Primary application domains"
)

# Add properties to concepts
ontology.set_concept_property(dl, "complexity", 0.8)
ontology.set_concept_property(dl, "application_domain", ["nlp", "computer_vision"])

ontology.set_concept_property(cnn, "complexity", 0.7)
ontology.set_concept_property(cnn, "application_domain", ["computer_vision"])
```

## Ontology Integration with SmartMemory

### Automatic Concept Extraction

Configure SmartMemory to use ontology for concept extraction:

```python
from smartmemory.smart_memory import SmartMemory

# Initialize SmartMemory with ontology
memory = SmartMemory(
    ontology_config={
        "enable_ontology": True,
        "ontology_manager": ontology,
        "auto_concept_extraction": True,
        "concept_confidence_threshold": 0.7
    }
)

# Add content with automatic concept extraction
content = """
Convolutional Neural Networks (CNNs) are a type of deep learning 
architecture particularly effective for computer vision tasks. 
They use convolution operations to detect local features in images.
"""

memory_id = memory.add(content, memory_type="semantic")

# SmartMemory automatically extracts and links concepts
extracted_concepts = memory.get_extracted_concepts(memory_id)
print(f"Extracted concepts: {extracted_concepts}")
# Output: ['ConvolutionalNeuralNetwork', 'DeepLearning', 'ComputerVision']
```

### Semantic Enhancement

Use ontology to enhance memory relationships:

```python
# Enhanced search using ontology
results = memory.search(
    "neural networks for image processing",
    use_ontology=True,
    expand_concepts=True,  # Include related concepts
    concept_expansion_depth=2
)

# Results will include memories about:
# - CNNs (direct match)
# - Deep Learning (parent concept)
# - Computer Vision (related domain)
# - Image Processing (related concept)
```

### Concept-Based Organization

Organize memories by ontological concepts:

```python
# Get memories by concept hierarchy
ai_memories = memory.get_memories_by_concept(
    "ArtificialIntelligence",
    include_subconcepts=True,  # Include ML, DL, etc.
    min_confidence=0.6
)

# Get concept statistics
concept_stats = memory.get_concept_statistics()
print(f"Most frequent concepts: {concept_stats['top_concepts']}")
print(f"Concept coverage: {concept_stats['coverage_percentage']}")
```

## Advanced Ontology Features

### Dynamic Ontology Evolution

Allow ontology to evolve based on content:

```python
class DynamicOntologyManager:
    def __init__(self, base_ontology):
        self.ontology = base_ontology
        self.concept_frequency = {}
        self.relationship_patterns = {}
    
    def analyze_content_patterns(self, memory):
        """Analyze content to discover new concepts and relationships."""
        
        # Extract co-occurrence patterns
        co_occurrences = memory.analyze_concept_co_occurrences()
        
        # Identify potential new concepts
        new_concepts = self.identify_emerging_concepts(co_occurrences)
        
        # Suggest new relationships
        new_relationships = self.suggest_relationships(co_occurrences)
        
        return {
            "new_concepts": new_concepts,
            "new_relationships": new_relationships,
            "confidence_scores": self.calculate_confidence_scores(
                new_concepts, new_relationships
            )
        }
    
    def evolve_ontology(self, suggestions, approval_threshold=0.8):
        """Automatically evolve ontology based on suggestions."""
        
        for concept in suggestions["new_concepts"]:
            if concept["confidence"] > approval_threshold:
                self.ontology.add_concept(
                    Concept(
                        concept["name"],
                        parent=concept["suggested_parent"],
                        confidence=concept["confidence"]
                    )
                )
        
        for relationship in suggestions["new_relationships"]:
            if relationship["confidence"] > approval_threshold:
                self.ontology.add_relationship(
                    Relationship(
                        relationship["source"],
                        relationship["type"],
                        relationship["target"],
                        strength=relationship["confidence"]
                    )
                )
```

### Multi-Domain Ontologies

Manage multiple domain-specific ontologies:

```python
class MultiDomainOntologyManager:
    def __init__(self):
        self.domains = {}
        self.cross_domain_mappings = {}
    
    def add_domain_ontology(self, domain_name, ontology):
        """Add a domain-specific ontology."""
        self.domains[domain_name] = ontology
    
    def create_cross_domain_mapping(self, domain1, concept1, domain2, concept2, mapping_type="EQUIVALENT"):
        """Create mappings between concepts in different domains."""
        mapping_key = f"{domain1}:{concept1} -> {domain2}:{concept2}"
        self.cross_domain_mappings[mapping_key] = {
            "type": mapping_type,
            "confidence": 0.9,
            "bidirectional": mapping_type == "EQUIVALENT"
        }
    
    def search_across_domains(self, query, target_domains=None):
        """Search across multiple domain ontologies."""
        results = {}
        
        domains_to_search = target_domains or self.domains.keys()
        
        for domain in domains_to_search:
            domain_results = self.domains[domain].search(query)
            
            # Apply cross-domain mappings
            enhanced_results = self.apply_cross_domain_mappings(
                domain_results, domain
            )
            
            results[domain] = enhanced_results
        
        return self.merge_cross_domain_results(results)

# Example: Technology and Business ontologies
tech_ontology = OntologyManager()
business_ontology = OntologyManager()

multi_domain = MultiDomainOntologyManager()
multi_domain.add_domain_ontology("technology", tech_ontology)
multi_domain.add_domain_ontology("business", business_ontology)

# Create cross-domain mappings
multi_domain.create_cross_domain_mapping(
    "technology", "MachineLearning",
    "business", "PredictiveAnalytics",
    "RELATED"
)
```

### Ontology Validation and Consistency

Ensure ontology consistency and quality:

```python
class OntologyValidator:
    def __init__(self, ontology):
        self.ontology = ontology
        self.validation_rules = []
    
    def add_validation_rule(self, rule):
        """Add custom validation rule."""
        self.validation_rules.append(rule)
    
    def validate_consistency(self):
        """Check ontology for consistency issues."""
        issues = []
        
        # Check for circular dependencies
        circular_deps = self.check_circular_dependencies()
        if circular_deps:
            issues.extend(circular_deps)
        
        # Check for orphaned concepts
        orphaned = self.check_orphaned_concepts()
        if orphaned:
            issues.extend(orphaned)
        
        # Check relationship constraints
        constraint_violations = self.check_relationship_constraints()
        if constraint_violations:
            issues.extend(constraint_violations)
        
        # Apply custom validation rules
        for rule in self.validation_rules:
            rule_issues = rule.validate(self.ontology)
            issues.extend(rule_issues)
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "severity_counts": self.count_by_severity(issues)
        }
    
    def check_circular_dependencies(self):
        """Detect circular inheritance chains."""
        visited = set()
        rec_stack = set()
        issues = []
        
        def has_cycle(concept):
            visited.add(concept.id)
            rec_stack.add(concept.id)
            
            for child in concept.children:
                if child.id not in visited:
                    if has_cycle(child):
                        return True
                elif child.id in rec_stack:
                    issues.append({
                        "type": "circular_dependency",
                        "severity": "error",
                        "message": f"Circular dependency detected: {concept.name} -> {child.name}",
                        "concepts": [concept.name, child.name]
                    })
                    return True
            
            rec_stack.remove(concept.id)
            return False
        
        for concept in self.ontology.get_root_concepts():
            if concept.id not in visited:
                has_cycle(concept)
        
        return issues
```

## Ontology-Driven Features

### Intelligent Query Expansion

Expand queries using ontological knowledge:

```python
class OntologyQueryExpander:
    def __init__(self, ontology):
        self.ontology = ontology
    
    def expand_query(self, query, expansion_strategy="hierarchical"):
        """Expand query using ontological relationships."""
        
        # Extract concepts from query
        query_concepts = self.extract_concepts_from_query(query)
        
        expanded_terms = set([query])
        
        for concept in query_concepts:
            if expansion_strategy == "hierarchical":
                # Add parent and child concepts
                parents = self.ontology.get_ancestors(concept)
                children = self.ontology.get_descendants(concept)
                expanded_terms.update([p.name for p in parents])
                expanded_terms.update([c.name for c in children])
            
            elif expansion_strategy == "semantic":
                # Add semantically related concepts
                related = self.ontology.get_related_concepts(
                    concept, 
                    relationship_types=["SIMILAR_TO", "RELATED_TO"]
                )
                expanded_terms.update([r.name for r in related])
            
            elif expansion_strategy == "domain":
                # Add concepts from same domain
                domain_concepts = self.ontology.get_concepts_by_domain(
                    concept.domain
                )
                expanded_terms.update([d.name for d in domain_concepts])
        
        return {
            "original_query": query,
            "expanded_terms": list(expanded_terms),
            "expansion_strategy": expansion_strategy
        }

# Usage with SmartMemory
expander = OntologyQueryExpander(ontology)

def enhanced_search(memory, query):
    # Expand query using ontology
    expansion = expander.expand_query(query, "hierarchical")
    
    # Search with expanded terms
    results = []
    for term in expansion["expanded_terms"]:
        term_results = memory.search(term, top_k=5)
        results.extend(term_results)
    
    # Deduplicate and rank results
    return memory.deduplicate_and_rank(results)
```

### Concept-Based Reasoning

Enable reasoning over ontological concepts:

```python
class OntologyReasoner:
    def __init__(self, ontology):
        self.ontology = ontology
        self.inference_rules = []
    
    def add_inference_rule(self, rule):
        """Add custom inference rule."""
        self.inference_rules.append(rule)
    
    def infer_relationships(self, concept1, concept2):
        """Infer possible relationships between concepts."""
        
        direct_relationships = self.ontology.get_relationships(concept1, concept2)
        
        # Infer through transitivity
        transitive_relationships = self.infer_transitive_relationships(
            concept1, concept2
        )
        
        # Apply custom inference rules
        inferred_relationships = []
        for rule in self.inference_rules:
            rule_inferences = rule.apply(concept1, concept2, self.ontology)
            inferred_relationships.extend(rule_inferences)
        
        return {
            "direct": direct_relationships,
            "transitive": transitive_relationships,
            "inferred": inferred_relationships
        }
    
    def explain_relationship(self, concept1, concept2, relationship_type):
        """Provide explanation for why two concepts are related."""
        
        # Find relationship path
        path = self.ontology.find_relationship_path(
            concept1, concept2, relationship_type
        )
        
        if not path:
            return None
        
        explanation = {
            "relationship": f"{concept1.name} {relationship_type} {concept2.name}",
            "path": [step.description for step in path],
            "confidence": min([step.confidence for step in path]),
            "reasoning": self.generate_reasoning_explanation(path)
        }
        
        return explanation

# Example inference rule
class DomainSimilarityRule:
    def apply(self, concept1, concept2, ontology):
        """Infer similarity based on shared domain."""
        
        domain1 = ontology.get_concept_domain(concept1)
        domain2 = ontology.get_concept_domain(concept2)
        
        if domain1 == domain2 and domain1 is not None:
            return [{
                "type": "SIMILAR_TO",
                "confidence": 0.6,
                "reasoning": f"Both concepts belong to {domain1} domain"
            }]
        
        return []
```

## Ontology Visualization and Management

### Ontology Visualization

Create visual representations of your ontology:

```python
import networkx as nx
import matplotlib.pyplot as plt

class OntologyVisualizer:
    def __init__(self, ontology):
        self.ontology = ontology
    
    def create_concept_graph(self):
        """Create NetworkX graph from ontology."""
        G = nx.DiGraph()
        
        # Add nodes (concepts)
        for concept in self.ontology.get_all_concepts():
            G.add_node(
                concept.id,
                label=concept.name,
                type="concept",
                properties=concept.properties
            )
        
        # Add edges (relationships)
        for relationship in self.ontology.get_all_relationships():
            G.add_edge(
                relationship.source.id,
                relationship.target.id,
                type=relationship.type,
                strength=relationship.strength
            )
        
        return G
    
    def visualize_hierarchy(self, root_concept=None, max_depth=3):
        """Visualize concept hierarchy."""
        G = self.create_concept_graph()
        
        if root_concept:
            # Create subgraph from root concept
            descendants = self.ontology.get_descendants(
                root_concept, max_depth=max_depth
            )
            subgraph_nodes = [root_concept.id] + [d.id for d in descendants]
            G = G.subgraph(subgraph_nodes)
        
        # Use hierarchical layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20)
        
        # Draw labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Ontology Concept Hierarchy")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def generate_ontology_report(self):
        """Generate comprehensive ontology report."""
        stats = self.ontology.get_statistics()
        
        report = f"""
# Ontology Report

## Overview
- **Total Concepts**: {stats['concept_count']}
- **Total Relationships**: {stats['relationship_count']}
- **Max Hierarchy Depth**: {stats['max_depth']}
- **Root Concepts**: {stats['root_concept_count']}

## Concept Distribution
"""
        
        for domain, count in stats['concepts_by_domain'].items():
            report += f"- **{domain}**: {count} concepts\n"
        
        report += f"""
## Relationship Types
"""
        
        for rel_type, count in stats['relationships_by_type'].items():
            report += f"- **{rel_type}**: {count} relationships\n"
        
        return report
```

### Ontology Import/Export

Support for standard ontology formats:

```python
class OntologySerializer:
    def __init__(self, ontology):
        self.ontology = ontology
    
    def export_to_owl(self, filename):
        """Export ontology to OWL format."""
        from rdflib import Graph, Namespace, RDF, RDFS, OWL
        
        g = Graph()
        
        # Define namespaces
        onto = Namespace("http://smartmemory.ai/ontology#")
        g.bind("onto", onto)
        
        # Add concepts as OWL classes
        for concept in self.ontology.get_all_concepts():
            concept_uri = onto[concept.name]
            g.add((concept_uri, RDF.type, OWL.Class))
            
            if concept.description:
                g.add((concept_uri, RDFS.comment, concept.description))
            
            if concept.parent:
                parent_uri = onto[concept.parent.name]
                g.add((concept_uri, RDFS.subClassOf, parent_uri))
        
        # Add relationships as object properties
        for relationship in self.ontology.get_all_relationships():
            prop_uri = onto[relationship.type]
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))
            
            source_uri = onto[relationship.source.name]
            target_uri = onto[relationship.target.name]
            g.add((source_uri, prop_uri, target_uri))
        
        # Serialize to file
        g.serialize(destination=filename, format='xml')
    
    def import_from_owl(self, filename):
        """Import ontology from OWL file."""
        from rdflib import Graph, RDF, RDFS, OWL
        
        g = Graph()
        g.parse(filename)
        
        # Extract concepts
        concepts = {}
        for subj, pred, obj in g.triples((None, RDF.type, OWL.Class)):
            concept_name = str(subj).split('#')[-1]
            concepts[concept_name] = Concept(concept_name)
        
        # Extract hierarchy
        for subj, pred, obj in g.triples((None, RDFS.subClassOf, None)):
            child_name = str(subj).split('#')[-1]
            parent_name = str(obj).split('#')[-1]
            
            if child_name in concepts and parent_name in concepts:
                concepts[child_name].parent = concepts[parent_name]
        
        # Add to ontology
        for concept in concepts.values():
            self.ontology.add_concept(concept)
```

## Best Practices

### Ontology Design Guidelines

1. **Start Simple**: Begin with core concepts and expand gradually
2. **Domain Focus**: Keep concepts relevant to your domain
3. **Consistent Naming**: Use clear, consistent naming conventions
4. **Balanced Hierarchy**: Avoid too deep or too shallow hierarchies
5. **Relationship Quality**: Define meaningful, well-documented relationships

### Performance Considerations

1. **Index Frequently Used Concepts**: Optimize for common queries
2. **Limit Expansion Depth**: Control query expansion to avoid performance issues
3. **Cache Ontology Queries**: Cache frequently accessed ontology information
4. **Batch Operations**: Use batch operations for bulk ontology updates

### Maintenance and Evolution

1. **Version Control**: Track ontology changes over time
2. **Validation**: Regularly validate ontology consistency
3. **User Feedback**: Incorporate user feedback for ontology improvements
4. **Automated Evolution**: Use content analysis to suggest ontology updates

## Next Steps

- **Advanced Features**: Explore [advanced features](advanced-features.md) for custom ontology algorithms
- **Performance Tuning**: Optimize ontology operations with [performance tuning](performance-tuning.md)
- **API Reference**: Complete API documentation in [SmartMemory API](../api/smart-memory.md)
- **Examples**: See ontology usage in [conversational AI examples](../examples/conversational-ai.md)

Effective ontology management enables SmartMemory to provide more intelligent, context-aware memory operations that understand the semantic structure of your knowledge domain.
