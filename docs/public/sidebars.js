/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a "Next" and "Previous" button
 - provide a way to navigate between docs

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
        'getting-started/configuration',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'concepts/overview',
        'concepts/memory-types',
        'concepts/zettelkasten-memory',
        'concepts/hybrid-storage',
        'concepts/ingestion-flow',
        'concepts/similarity-framework',
        {
          type: 'category',
          label: 'Evolution Algorithms',
          items: [
            'concepts/evolution-algorithms',
            {
              type: 'category',
              label: 'Core Evolvers',
              items: [
                'concepts/evolution-algorithms/working-to-episodic',
                'concepts/evolution-algorithms/episodic-to-semantic',
                'concepts/evolution-algorithms/episodic-to-zettel',
                'concepts/evolution-algorithms/working-to-procedural',
                'concepts/evolution-algorithms/episodic-decay',
                'concepts/evolution-algorithms/semantic-decay',
                'concepts/evolution-algorithms/zettel-pruning',
              ],
            },
            {
              type: 'category',
              label: 'Enhanced Algorithms',
              items: [
                'concepts/evolution-algorithms/exponential-decay',
                'concepts/evolution-algorithms/retrieval-strengthening',
                'concepts/evolution-algorithms/interference-consolidation',
              ],
            },
            {
              type: 'category',
              label: 'Agent-Optimized',
              items: [
                'concepts/evolution-algorithms/maximal-connectivity',
                'concepts/evolution-algorithms/strategic-pruning',
                'concepts/evolution-algorithms/rapid-enrichment',
                'concepts/evolution-algorithms/hierarchical-organization',
              ],
            },
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/basic-usage',
        'guides/mcp-integration',
        'guides/advanced-features',
        'guides/performance-tuning',
        'guides/ontology-management',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/smart-memory',
        'api/memory-types',
        'api/components',
        'api/factories',
        'api/tools',
      ],
    },
    {
      type: 'category',
      label: 'Examples',
      items: [
        'examples/conversational-ai',
        'examples/learning-assistant',
        'examples/knowledge-graph',
      ],
    },
  ],
};

module.exports = sidebars;
