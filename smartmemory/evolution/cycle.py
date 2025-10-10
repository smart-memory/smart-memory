from smartmemory.plugins.manager import get_plugin_manager


def run_evolution_cycle(memory, config=None, logger=None, evolver_names=None):
    """
    Runs evolvers in sequence using the plugin system.
    
    Args:
        memory: The memory system to evolve
        config: Optional config dict or typed config object
        logger: Optional logger for tracking evolution
        evolver_names: Optional list of evolver names to run. If None, runs all registered evolvers.
    
    The function now uses the PluginRegistry to discover and load evolvers,
    supporting both built-in and external plugins.
    """
    # Get plugin manager and registry
    plugin_manager = get_plugin_manager()
    registry = plugin_manager.registry
    
    # Get evolver names to run
    if evolver_names is None:
        evolver_names = registry.list_plugins('evolver')
    
    # Run each evolver
    for evolver_name in evolver_names:
        evolver_class = registry.get_evolver(evolver_name)
        if evolver_class:
            try:
                evolver = evolver_class(config=config or {})
                evolver.evolve(memory, logger=logger)
            except Exception as e:
                if logger:
                    logger.error(f"Error running evolver '{evolver_name}': {e}")
                # Continue with other evolvers even if one fails
