"""
SmartMemory CLI

Command-line interface for SmartMemory operations.
Only available when installed with: pip install smartmemory[cli]
"""

def main():
    """Main CLI entry point with lazy imports."""
    try:
        import click
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print("❌ CLI dependencies not installed.")
        print("Install with: pip install smartmemory[cli]")
        return 1
    
    console = Console()
    
    @click.group()
    @click.version_option()
    def cli():
        """SmartMemory - Multi-Layered AI Memory System CLI"""
        pass
    
    @cli.command()
    def info():
        """Show SmartMemory version and configuration"""
        from smartmemory import __version__
        
        console.print(f"[bold cyan]SmartMemory v{__version__}[/bold cyan]")
        console.print("Multi-Layered AI Memory System")
        console.print()
        console.print("📦 Installation: [green]OK[/green]")
        
        # Check optional dependencies
        deps = {
            "chromadb": "Vector database",
            "falkordb": "Graph database",
            "spacy": "NLP processing",
            "litellm": "LLM integration"
        }
        
        console.print("\n[bold]Core Dependencies:[/bold]")
        for dep, desc in deps.items():
            try:
                __import__(dep)
                console.print(f"  ✅ {dep}: {desc}")
            except ImportError:
                console.print(f"  ❌ {dep}: {desc} [red](not installed)[/red]")
    
    @cli.group()
    def plugins():
        """Plugin management commands"""
        pass
    
    @plugins.command("list")
    @click.option('--type', '-t', 'plugin_type', 
                  type=click.Choice(['enricher', 'evolver', 'extractor', 'grounder', 'all']),
                  default='all',
                  help='Filter by plugin type')
    def list_plugins(plugin_type):
        """List all installed plugins"""
        from smartmemory.plugins.manager import get_plugin_manager
        
        manager = get_plugin_manager()
        
        if plugin_type == 'all':
            types = ['enricher', 'evolver', 'extractor', 'grounder']
        else:
            types = [plugin_type]
        
        for ptype in types:
            plugins_list = manager.registry.list_plugins(ptype)
            
            if plugins_list:
                table = Table(title=f"{ptype.capitalize()}s ({len(plugins_list)})")
                table.add_column("Name", style="cyan")
                table.add_column("Version", style="green")
                table.add_column("Description", style="white")
                
                for plugin_name in sorted(plugins_list):
                    metadata = manager.registry.get_metadata(plugin_name)
                    if metadata:
                        table.add_row(
                            metadata.name,
                            metadata.version,
                            metadata.description[:60] + "..." if len(metadata.description) > 60 else metadata.description
                        )
                
                console.print(table)
                console.print()
    
    @plugins.command("info")
    @click.argument('plugin_name')
    def plugin_info(plugin_name):
        """Show detailed information about a plugin"""
        from smartmemory.plugins.manager import get_plugin_manager
        
        manager = get_plugin_manager()
        metadata = manager.registry.get_metadata(plugin_name)
        
        if not metadata:
            console.print(f"[red]❌ Plugin '{plugin_name}' not found[/red]")
            return 1
        
        console.print(f"\n[bold cyan]{metadata.name}[/bold cyan] v{metadata.version}")
        console.print(f"[dim]{metadata.description}[/dim]")
        console.print()
        console.print(f"[bold]Type:[/bold] {metadata.plugin_type}")
        console.print(f"[bold]Author:[/bold] {metadata.author}")
        
        if metadata.dependencies:
            console.print(f"[bold]Dependencies:[/bold]")
            for dep in metadata.dependencies:
                console.print(f"  • {dep}")
        
        if metadata.tags:
            console.print(f"[bold]Tags:[/bold] {', '.join(metadata.tags)}")
    
    @cli.command()
    @click.argument('text')
    @click.option('--memory-type', '-t', default='semantic', 
                  type=click.Choice(['working', 'semantic', 'episodic', 'procedural', 'zettel']),
                  help='Memory type')
    @click.option('--user-id', '-u', help='User ID')
    def add(text, memory_type, user_id):
        """Add a memory item"""
        from smartmemory import SmartMemory, MemoryItem
        
        try:
            memory = SmartMemory()
            item = MemoryItem(
                content=text,
                memory_type=memory_type,
                user_id=user_id
            )
            
            result = memory.add(item)
            
            if result:
                console.print(f"[green]✅ Memory added successfully![/green]")
                console.print(f"ID: {result.item_id}")
                console.print(f"Type: {result.memory_type}")
            else:
                console.print("[red]❌ Failed to add memory[/red]")
                return 1
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1
    
    @cli.command()
    @click.argument('query')
    @click.option('--top-k', '-k', default=5, help='Number of results')
    @click.option('--user-id', '-u', help='User ID')
    def search(query, top_k, user_id):
        """Search memories"""
        from smartmemory import SmartMemory
        
        try:
            memory = SmartMemory()
            results = memory.search(query, top_k=top_k)
            
            if not results:
                console.print("[yellow]No results found[/yellow]")
                return
            
            console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")
            
            for i, item in enumerate(results, 1):
                console.print(f"[cyan]{i}. {item.item_id}[/cyan]")
                console.print(f"   Type: {item.memory_type}")
                console.print(f"   Content: {item.content[:100]}...")
                if item.metadata:
                    console.print(f"   Metadata: {item.metadata}")
                console.print()
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1
    
    @cli.command()
    def summary():
        """Show memory system summary"""
        from smartmemory import SmartMemory
        
        try:
            memory = SmartMemory()
            summary_data = memory.summary()
            
            console.print("\n[bold cyan]Memory System Summary[/bold cyan]\n")
            
            if isinstance(summary_data, dict):
                for key, value in summary_data.items():
                    console.print(f"[bold]{key}:[/bold] {value}")
            else:
                console.print(f"Total memories: {summary_data}")
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1
    
    @cli.group()
    def zettel():
        """Zettelkasten note-taking commands"""
        pass
    
    @zettel.command("add")
    @click.argument('content')
    @click.option('--title', '-t', help='Note title')
    @click.option('--tags', '-g', multiple=True, help='Tags (can specify multiple)')
    @click.option('--concepts', '-c', multiple=True, help='Concepts (can specify multiple)')
    @click.option('--note-id', '-i', help='Custom note ID')
    def zettel_add(content, title, tags, concepts, note_id):
        """Add a Zettelkasten note"""
        from smartmemory.memory.types.zettel_memory import ZettelMemory
        from smartmemory.models.memory_item import MemoryItem
        
        try:
            zettel_mem = ZettelMemory()
            
            metadata = {}
            if title:
                metadata['title'] = title
            if tags:
                metadata['tags'] = list(tags)
            if concepts:
                metadata['concepts'] = list(concepts)
            
            item = MemoryItem(
                content=content,
                metadata=metadata,
                item_id=note_id
            )
            
            result = zettel_mem.add(item)
            
            if result:
                console.print(f"[green]✅ Zettel note added successfully![/green]")
                console.print(f"ID: {result.item_id}")
                console.print(f"Title: {result.metadata.get('title', 'N/A')}")
                
                # Show parsed wikilinks if any
                if 'wikilinks' in result.metadata and result.metadata['wikilinks']:
                    console.print(f"Wikilinks: {', '.join(result.metadata['wikilinks'])}")
            else:
                console.print("[red]❌ Failed to add note[/red]")
                return 1
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1
    
    @zettel.command("overview")
    def zettel_overview():
        """Show Zettelkasten system overview"""
        from smartmemory.memory.types.zettel_memory import ZettelMemory
        
        try:
            zettel_mem = ZettelMemory()
            overview = zettel_mem.get_zettelkasten_overview()
            
            console.print("\n[bold cyan]📊 Zettelkasten Overview[/bold cyan]\n")
            
            console.print(f"[bold]Total notes:[/bold] {overview.get('total_notes', 0)}")
            console.print(f"[bold]Total connections:[/bold] {overview.get('total_connections', 0)}")
            console.print(f"[bold]Connection density:[/bold] {overview.get('connection_density', 0):.4f}")
            console.print(f"[bold]Knowledge clusters:[/bold] {overview.get('knowledge_clusters', 0)}")
            console.print(f"[bold]System health:[/bold] {overview.get('system_health', 'unknown').upper()}")
            console.print(f"[bold]Auto-linking:[/bold] {'✅ Enabled' if overview.get('auto_linking_enabled') else '❌ Disabled'}")
            
            # Show top clusters
            top_clusters = overview.get('top_clusters', [])
            if top_clusters:
                console.print("\n[bold]🏆 Top Knowledge Clusters:[/bold]")
                for cluster in top_clusters[:3]:
                    console.print(f"  • {cluster.get('id')}: {cluster.get('size')} notes")
                    console.print(f"    Concepts: {', '.join(cluster.get('concepts', [])[:3])}")
            
            # Show emerging concepts
            emerging = overview.get('emerging_concepts', {})
            if emerging:
                console.print("\n[bold]🌱 Top Emerging Concepts:[/bold]")
                for concept, score in list(emerging.items())[:5]:
                    console.print(f"  • {concept}: {score:.3f}")
                    
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1
    
    @zettel.command("backlinks")
    @click.argument('note_id')
    def zettel_backlinks(note_id):
        """Show backlinks for a note"""
        from smartmemory.memory.types.zettel_memory import ZettelMemory
        
        try:
            zettel_mem = ZettelMemory()
            backlinks = zettel_mem.get_backlinks(note_id)
            
            if not backlinks:
                console.print(f"[yellow]No backlinks found for '{note_id}'[/yellow]")
                return
            
            console.print(f"\n[bold cyan]📥 Backlinks for '{note_id}' ({len(backlinks)})[/bold cyan]\n")
            
            for note in backlinks:
                console.print(f"[cyan]← {note.item_id}[/cyan]")
                console.print(f"  Title: {note.metadata.get('title', 'N/A')}")
                console.print(f"  Content: {note.content[:80]}...")
                console.print()
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1
    
    @zettel.command("connections")
    @click.argument('note_id')
    def zettel_connections(note_id):
        """Show all connections for a note"""
        from smartmemory.memory.types.zettel_memory import ZettelMemory
        
        try:
            zettel_mem = ZettelMemory()
            connections = zettel_mem.get_bidirectional_connections(note_id)
            
            console.print(f"\n[bold cyan]🔗 Connections for '{note_id}'[/bold cyan]\n")
            
            console.print(f"[bold]Forward links ({len(connections['forward_links'])}):[/bold]")
            for note in connections['forward_links'][:5]:
                console.print(f"  → {note.item_id}: {note.metadata.get('title', 'N/A')}")
            
            console.print(f"\n[bold]Backlinks ({len(connections['backlinks'])}):[/bold]")
            for note in connections['backlinks'][:5]:
                console.print(f"  ← {note.item_id}: {note.metadata.get('title', 'N/A')}")
            
            console.print(f"\n[bold]Tag-related ({len(connections['related_by_tags'])}):[/bold]")
            for note in connections['related_by_tags'][:5]:
                console.print(f"  # {note.item_id}: {note.metadata.get('title', 'N/A')}")
            
            console.print(f"\n[bold]Concept-related ({len(connections['related_by_concepts'])}):[/bold]")
            for note in connections['related_by_concepts'][:5]:
                console.print(f"  (( {note.item_id}: {note.metadata.get('title', 'N/A')}")
            
            total = sum(len(conn_list) for conn_list in connections.values())
            console.print(f"\n[bold]Total connections: {total}[/bold]")
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1
    
    @zettel.command("suggest")
    @click.argument('note_id')
    @click.option('--count', '-n', default=5, help='Number of suggestions')
    def zettel_suggest(note_id, count):
        """Suggest related notes"""
        from smartmemory.memory.types.zettel_memory import ZettelMemory
        
        try:
            zettel_mem = ZettelMemory()
            suggestions = zettel_mem.suggest_related_notes(note_id, suggestion_count=count)
            
            if not suggestions:
                console.print(f"[yellow]No suggestions found for '{note_id}'[/yellow]")
                return
            
            console.print(f"\n[bold cyan]💡 Related Note Suggestions for '{note_id}'[/bold cyan]\n")
            
            for i, (note, score, reason) in enumerate(suggestions, 1):
                console.print(f"[bold]{i}. {note.item_id}[/bold]")
                console.print(f"   Title: {note.metadata.get('title', 'N/A')}")
                console.print(f"   Relevance: {score:.2f}")
                console.print(f"   Reason: {reason}")
                console.print()
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1
    
    @zettel.command("clusters")
    @click.option('--min-size', '-m', default=3, help='Minimum cluster size')
    def zettel_clusters(min_size):
        """Detect knowledge clusters"""
        from smartmemory.memory.types.zettel_memory import ZettelMemory
        
        try:
            zettel_mem = ZettelMemory()
            clusters = zettel_mem.detect_knowledge_clusters(min_cluster_size=min_size)
            
            if not clusters:
                console.print(f"[yellow]No clusters found (min size: {min_size})[/yellow]")
                return
            
            console.print(f"\n[bold cyan]🌐 Knowledge Clusters ({len(clusters)})[/bold cyan]\n")
            
            for i, cluster in enumerate(clusters[:10], 1):
                console.print(f"[bold]{i}. {cluster.cluster_id}[/bold]")
                console.print(f"   Notes: {len(cluster.note_ids)}")
                console.print(f"   Concepts: {', '.join(cluster.central_concepts[:5])}")
                console.print(f"   Density: {cluster.connection_density:.3f}")
                console.print(f"   Emergence score: {cluster.emergence_score:.2f}")
                console.print()
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1
    
    @zettel.command("parse")
    @click.argument('content')
    def zettel_parse(content):
        """Parse wikilinks from content"""
        from smartmemory.memory.types.zettel_memory import ZettelMemory
        
        try:
            zettel_mem = ZettelMemory()
            parsed = zettel_mem.parse_wikilinks(content)
            
            console.print("\n[bold cyan]📝 Parsed Links[/bold cyan]\n")
            
            if parsed['wikilinks']:
                console.print(f"[bold]Wikilinks ({len(parsed['wikilinks'])}):[/bold]")
                for link in parsed['wikilinks']:
                    console.print(f"  → [[{link}]]")
            
            if parsed['concepts']:
                console.print(f"\n[bold]Concepts ({len(parsed['concepts'])}):[/bold]")
                for concept in parsed['concepts']:
                    console.print(f"  → (({concept}))")
            
            if parsed['hashtags']:
                console.print(f"\n[bold]Hashtags ({len(parsed['hashtags'])}):[/bold]")
                for tag in parsed['hashtags']:
                    console.print(f"  → #{tag}")
            
            if not any(parsed.values()):
                console.print("[yellow]No links found in content[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return 1
    
    # Run the CLI
    return cli()


if __name__ == "__main__":
    exit(main())
