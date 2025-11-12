import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

// Comprehensive feature catalog organized by category
const CoreMemoryFeatures = [
  {
    title: '🧠 Semantic Memory',
    description: 'Facts, concepts, and general knowledge with automatic entity extraction and relationship mapping'
  },
  {
    title: '📚 Episodic Memory', 
    description: 'Personal experiences and events with temporal context and emotional metadata'
  },
  {
    title: '⚙️ Procedural Memory',
    description: 'Skills, procedures, and how-to knowledge with step-by-step process tracking'
  },
  {
    title: '💭 Working Memory',
    description: 'Active context and immediate focus with adaptive capacity management'
  },
  {
    title: '🗂️ Zettelkasten Memory',
    description: 'Atomic knowledge notes with bidirectional linking and emergent knowledge graphs'
  }
];

const ProcessingFeatures = [
  {
    title: '🔍 Entity Extraction',
    description: 'Automatic identification of people, places, concepts, and relationships using NLP'
  },
  {
    title: '🔗 Intelligent Linking',
    description: 'Automatic relationship discovery through semantic similarity and entity overlap'
  },
  {
    title: '🎯 Grounding & Provenance',
    description: 'Source attribution and fact verification with audit trails and confidence scoring'
  },
  {
    title: '📊 Semantic Analysis',
    description: 'Deep content understanding with contextual classification and enrichment'
  },
  {
    title: '⏰ Background Processing',
    description: 'Asynchronous ingestion and evolution with configurable worker pools'
  }
];

const EvolutionFeatures = [
  {
    title: '🧬 Working→Episodic',
    description: 'Automatic consolidation of working memory into episodic experiences'
  },
  {
    title: '📈 Episodic→Semantic', 
    description: 'Knowledge extraction from experiences with pattern recognition'
  },
  {
    title: '🗂️ Episodic→Zettelkasten',
    description: 'Atomic note creation from episodic events with concept linking'
  },
  {
    title: '✂️ Memory Pruning',
    description: 'Intelligent cleanup of duplicate, low-quality, or outdated memories'
  },
  {
    title: '💪 Retrieval Strengthening',
    description: 'Access-based memory reinforcement with decay algorithms'
  },
  {
    title: '🎯 Strategic Optimization',
    description: 'AI-driven memory organization with maximal connectivity and hierarchical structure'
  }
];

const SearchFeatures = [
  {
    title: '🔍 Semantic Search',
    description: 'Vector-based similarity search across all memory types with relevance scoring'
  },
  {
    title: '🎯 Multi-Modal Queries',
    description: 'Search by content, metadata, temporal range, user context, and confidence levels'
  },
  {
    title: '🕸️ Graph Traversal',
    description: 'Relationship-based discovery with path finding and neighborhood exploration'
  },
  {
    title: '👤 User-Specific Search',
    description: 'Personalized results with user isolation and preference learning'
  },
  {
    title: '⚡ Real-Time Results',
    description: 'Sub-millisecond response times with intelligent caching and indexing'
  }
];

const IntegrationFeatures = [
  {
    title: '🔧 MCP Protocol',
    description: 'Model Context Protocol tools for seamless AI agent integration'
  },
  {
    title: '🤖 LangChain Support',
    description: 'Native integration with LangChain agents and tool ecosystems'
  },
  {
    title: '⚡ AutoGen Compatible',
    description: 'Multi-agent conversation support with shared memory contexts'
  },
  {
    title: '🔌 Custom Similarity',
    description: 'Pluggable similarity metrics for domain-specific applications'
  },
  {
    title: '📡 REST API',
    description: 'HTTP endpoints for language-agnostic integration and web applications'
  }
];

const StorageFeatures = [
  {
    title: '📊 Hybrid Architecture',
    description: 'Graph + Vector + Metadata triple storage for optimal performance'
  },
  {
    title: '🗃️ Multiple Backends',
    description: 'Support for FalkorDB, Neo4j, ChromaDB, Pinecone, and custom implementations'
  },
  {
    title: '📈 Auto-Scaling',
    description: 'Dynamic capacity adjustment with load balancing and horizontal scaling'
  },
  {
    title: '🔒 Multi-Tenancy',
    description: 'Secure user isolation with namespace partitioning and permission controls'
  },
  {
    title: '💾 Persistence Layer',
    description: 'Durable storage with backup, archival, and lifecycle management'
  }
];

const DeveloperFeatures = [
  {
    title: '📖 Comprehensive API',
    description: 'Full CRUD operations with batch processing and streaming support'
  },
  {
    title: '🔧 Configuration System',
    description: 'Flexible JSON/YAML configuration with environment variable support'
  },
  {
    title: '📊 Analytics & Monitoring',
    description: 'Performance metrics, health checks, and background processing statistics'
  },
  {
    title: '🧪 Testing Framework',
    description: 'Comprehensive test suite with mocking, fixtures, and performance benchmarks'
  },
  {
    title: '📚 Rich Documentation',
    description: 'Examples, tutorials, API reference, and integration guides'
  }
];

const FeatureList = [
  ...CoreMemoryFeatures,
  ...ProcessingFeatures, 
  ...EvolutionFeatures,
  ...SearchFeatures,
  ...IntegrationFeatures,
  ...StorageFeatures,
  ...DeveloperFeatures
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--6 col--lg-4')}>
      <div className="text--center padding-horiz--md margin-bottom--lg">
        <h4>{title}</h4>
        <p>{description}</p>
      </div>
    </div>
  );
}

function FeatureSection({title, features, id}) {
  return (
    <section className="margin-bottom--xl" id={id}>
      <div className="container">
        <div className="text--center margin-bottom--lg">
          <h2>{title}</h2>
        </div>
        <div className="row">
          {features.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

export default function HomepageFeatures() {
  return (
    <div className={styles.features}>
      <div className="container margin-bottom--xl">
        <div className="text--center">
          <h1>Complete Feature Reference</h1>
          <p className="margin-bottom--lg">SmartMemory provides comprehensive memory capabilities for AI agents and intelligent applications. Explore all features organized by category.</p>
          
          <div className="margin-bottom--lg">
            <a href="#core-memory" className="button button--outline button--sm margin-horiz--sm">Memory Types</a>
            <a href="#processing" className="button button--outline button--sm margin-horiz--sm">Processing</a>
            <a href="#evolution" className="button button--outline button--sm margin-horiz--sm">Evolution</a>
            <a href="#search" className="button button--outline button--sm margin-horiz--sm">Search</a>
            <a href="#integration" className="button button--outline button--sm margin-horiz--sm">Integration</a>
            <a href="#storage" className="button button--outline button--sm margin-horiz--sm">Storage</a>
            <a href="#developer" className="button button--outline button--sm margin-horiz--sm">Developer</a>
          </div>
          
          <div className="margin-bottom--xl">
            <Link to="/docs/getting-started/installation" className="button button--primary button--lg margin-horiz--md">
              📚 Get Started
            </Link>
            <Link to="/docs/getting-started/quick-start" className="button button--secondary button--lg margin-horiz--md">
              🚀 Quick Start Guide
            </Link>
            <Link to="/docs/api/smart-memory" className="button button--outline button--secondary button--lg margin-horiz--md">
              📖 API Reference
            </Link>
          </div>
        </div>
      </div>
      
      <FeatureSection 
        id="core-memory"
        title="🧠 Core Memory Types" 
        features={CoreMemoryFeatures} 
      />
      
      <FeatureSection 
        id="processing"
        title="⚙️ Intelligent Processing" 
        features={ProcessingFeatures} 
      />
      
      <FeatureSection 
        id="evolution"
        title="🧬 Evolution Algorithms" 
        features={EvolutionFeatures} 
      />
      
      <FeatureSection 
        id="search"
        title="🔍 Advanced Search" 
        features={SearchFeatures} 
      />
      
      <FeatureSection 
        id="integration"
        title="🔌 AI Agent Integration" 
        features={IntegrationFeatures} 
      />
      
      <FeatureSection 
        id="storage"
        title="💾 Storage & Architecture" 
        features={StorageFeatures} 
      />
      
      <FeatureSection 
        id="developer"
        title="🛠️ Developer Experience" 
        features={DeveloperFeatures} 
      />
      
      {/* Competitive Comparison Table */}
      <div className={styles.competitiveSection}>
        <div className="container">
          <h2 className={styles.competitiveTitle}>🏆 Why Choose SmartMemory?</h2>
          <p className={styles.competitiveSubtitle}>Compare leading agentic memory systems</p>
          
          <div className={styles.competitiveTable}>
            <div className={styles.competitiveColumn}>
              <div className={styles.competitiveHeader + ' ' + styles.smartmemory}>
                <h3>🧠 SmartMemory</h3>
                <p>Most Advanced</p>
              </div>
              <div className={styles.competitiveFeatures}>
                <div className={styles.feature}>✅ 5 Memory Types</div>
                <div className={styles.feature}>✅ 14+ Evolution Algorithms</div>
                <div className={styles.feature}>✅ Hybrid Storage (Graph+Vector)</div>
                <div className={styles.feature}>✅ Advanced Entity Extraction</div>
                <div className={styles.feature}>✅ Full MCP Protocol</div>
                <div className={styles.feature}>✅ Grounding & Provenance</div>
                <div className={styles.feature}>✅ Multi-Agent Support</div>
                <div className={styles.feature}>✅ Enterprise Features</div>
                <div className={styles.feature}>✅ Background Processing</div>
                <div className={styles.feature}>✅ Graph Traversal</div>
              </div>
              <div className={styles.competitiveUseCase}>
                <strong>Best for:</strong> Complex AI agents, enterprise apps, research systems
              </div>
            </div>
            
            <div className={styles.competitiveColumn}>
              <div className={styles.competitiveHeader + ' ' + styles.reg}>
                <h3>🔧 Zep</h3>
                <p>Lightweight</p>
              </div>
              <div className={styles.competitiveFeatures}>
                <div className={styles.feature}>✅ Basic Vector Search</div>
                <div className={styles.feature}>✅ Simple CRUD</div>
                <div className={styles.feature}>✅ Fast Performance</div>
                <div className={styles.feature}>❌ Limited Memory Types</div>
                <div className={styles.feature}>❌ No Evolution</div>
                <div className={styles.feature}>❌ No Grounding</div>
                <div className={styles.feature}>❌ Basic Integration</div>
                <div className={styles.feature}>❌ No Enterprise Features</div>
                <div className={styles.feature}>❌ No Background Processing</div>
                <div className={styles.feature}>❌ No Graph Support</div>
              </div>
              <div className={styles.competitiveUseCase}>
                <strong>Best for:</strong> Simple chatbots, basic RAG, proof-of-concepts
              </div>
            </div>
            
            <div className={styles.competitiveColumn}>
              <div className={styles.competitiveHeader + ' ' + styles.mem0}>
                <h3>☁️ Mem0</h3>
                <p>Cloud-Native</p>
              </div>
              <div className={styles.competitiveFeatures}>
                <div className={styles.feature}>✅ Good LLM Integration</div>
                <div className={styles.feature}>✅ Cloud-First</div>
                <div className={styles.feature}>✅ User Isolation</div>
                <div className={styles.feature}>⚠️ Basic Memory Types</div>
                <div className={styles.feature}>❌ No Evolution</div>
                <div className={styles.feature}>❌ Limited Grounding</div>
                <div className={styles.feature}>⚠️ Basic Agent Support</div>
                <div className={styles.feature}>⚠️ Basic Enterprise</div>
                <div className={styles.feature}>⚠️ Basic Processing</div>
                <div className={styles.feature}>❌ Limited Graph</div>
              </div>
              <div className={styles.competitiveUseCase}>
                <strong>Best for:</strong> Cloud apps, basic personalization, simple memory needs
              </div>
            </div>
          </div>
          
          <div className={styles.competitiveCta}>
            <Link
              className="button button--primary button--lg"
              to="/docs/getting-started/installation">
              🚀 Get Started with SmartMemory
            </Link>
            <Link
              className="button button--outline button--secondary button--lg margin-left--md"
              to="/docs/intro#smartmemory-vs-competitors">
              📊 View Full Comparison
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
