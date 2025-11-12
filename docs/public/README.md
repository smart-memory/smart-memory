# SmartMemory Documentation

This directory contains the complete Docusaurus documentation for the SmartMemory agentic memory system.

## Quick Start

### Prerequisites

- Node.js 16.14 or higher
- npm or yarn

### Installation

```bash
cd docs/public
npm install
```

### Development

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

```bash
npm run deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.

## Documentation Structure

```
docs/
├── intro.md                    # Main introduction
├── getting-started/
│   ├── installation.md         # Installation guide
│   ├── quick-start.md          # Quick start tutorial
│   └── configuration.md        # Configuration options
├── concepts/
│   ├── overview.md             # Core concepts overview
│   ├── memory-types.md         # Memory type details
│   ├── ingestion-flow.md       # Processing pipeline
│   ├── evolution-algorithms.md # Memory evolution
│   └── similarity-framework.md # Similarity metrics
├── architecture/
│   ├── system-overview.md      # System architecture
│   ├── smart-memory.md         # SmartMemory class
│   ├── graph-backend.md        # Graph database layer
│   ├── components.md           # Component architecture
│   └── background-processing.md # Background processing
├── api/
│   ├── smart-memory.md         # SmartMemory API
│   ├── memory-types.md         # Memory type APIs
│   ├── factories.md            # Factory pattern APIs
│   ├── components.md           # Component APIs
│   └── tools.md                # MCP tools and integrations
├── guides/
│   ├── basic-usage.md          # Basic usage patterns
│   ├── advanced-features.md    # Advanced capabilities
│   ├── ontology-management.md  # Ontology system
│   ├── mcp-integration.md      # MCP integration
│   ├── background-processing.md # Background processing
│   └── performance-tuning.md   # Performance optimization
├── examples/
│   ├── simple-memory.md        # Simple examples
│   ├── conversational-ai.md    # Conversational AI
│   ├── learning-assistant.md   # Learning systems
│   └── knowledge-graph.md      # Knowledge graphs
├── advanced/
│   ├── custom-evolvers.md      # Custom evolution algorithms
│   ├── similarity-metrics.md   # Custom similarity metrics
│   ├── graph-operations.md     # Graph operations
│   ├── benchmarking.md         # Performance benchmarking
│   └── debugging.md            # Debugging guide
└── development/
    ├── contributing.md         # Contribution guidelines
    ├── testing.md              # Testing guide
    └── architecture-decisions.md # ADRs
```

## Writing Guidelines

### Style Guide

1. **Use clear, concise language**
2. **Include practical examples** for all concepts
3. **Provide complete code samples** that can be run
4. **Use consistent terminology** throughout
5. **Include error handling** in examples
6. **Add performance considerations** where relevant

### Code Examples

- All Python code should be complete and runnable
- Include necessary imports
- Use meaningful variable names
- Add comments for complex logic
- Show both basic and advanced usage patterns

### API Documentation

- Include method signatures with type hints
- Document all parameters and return values
- Provide usage examples for each method
- Include error handling information
- Show integration patterns

## Content Guidelines

### Target Audience

- **Primary**: Python developers building AI applications
- **Secondary**: Data scientists and ML engineers
- **Tertiary**: System architects and DevOps engineers

### Content Types

1. **Tutorials**: Step-by-step guides for specific tasks
2. **How-to Guides**: Solutions to common problems
3. **Reference**: Comprehensive API documentation
4. **Explanations**: Conceptual overviews and theory

### Writing Checklist

- [ ] Clear introduction explaining the purpose
- [ ] Prerequisites and requirements listed
- [ ] Step-by-step instructions with code examples
- [ ] Expected outputs shown
- [ ] Common errors and troubleshooting
- [ ] Next steps and related topics
- [ ] Links to relevant documentation

## Maintenance

### Regular Updates

- Update version numbers when releases occur
- Review and update code examples for compatibility
- Add new features and capabilities as they're developed
- Update performance benchmarks and metrics
- Refresh screenshots and diagrams

### Quality Assurance

- Test all code examples before publishing
- Verify all links work correctly
- Check for consistent terminology
- Ensure proper formatting and styling
- Review for accessibility compliance

## Contributing

### Adding New Documentation

1. Create a new markdown file in the appropriate directory
2. Add the file to `sidebars.js` in the correct location
3. Follow the established style guide
4. Include practical examples
5. Test all code samples
6. Submit a pull request

### Updating Existing Documentation

1. Make changes to the relevant markdown files
2. Update any affected navigation or links
3. Test changes locally with `npm start`
4. Verify all examples still work
5. Submit a pull request with clear description

### Documentation Review Process

1. Technical accuracy review
2. Style and consistency check
3. Code example testing
4. Accessibility review
5. Final approval and merge

## Technical Details

### Docusaurus Configuration

- **Version**: 2.4.3
- **Theme**: Classic preset
- **Plugins**: Standard documentation plugins
- **Styling**: Custom CSS with SmartMemory branding

### Build Process

1. Markdown files are processed by Docusaurus
2. Code examples are syntax highlighted
3. Navigation is generated from `sidebars.js`
4. Static site is built to `build/` directory
5. Can be deployed to any static hosting service

### Search Integration

- Built-in search functionality
- Indexes all documentation content
- Supports fuzzy matching
- Keyboard shortcuts for quick access

## Support

For questions about the documentation:

1. Check existing documentation first
2. Search for related issues on GitHub
3. Create a new issue with the `documentation` label
4. Provide specific details about what's unclear

For technical support with SmartMemory itself:

1. Refer to the main project README
2. Check the troubleshooting guides
3. Search existing GitHub issues
4. Create a new issue with appropriate labels

## License

This documentation is part of the SmartMemory project and follows the same license terms.
