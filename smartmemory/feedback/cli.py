def human_feedback_fn(step, context, results):
    """
    Example function for human-in-the-loop feedback via CLI. Replace with real UI or CLI prompt as needed.
    """
    print(f"Step {step}: Retrieved {len(results)} results.")
    user_input = input("Provide feedback (or type 'stop' to halt): ")
    return user_input
