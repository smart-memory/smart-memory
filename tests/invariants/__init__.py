"""
Invariant tests for SmartMemory core library.

Per testing philosophy, these test pure logic kernels that:
- Are expensive to trigger via integration tests
- Require specific edge case combinations
- Test permission evaluation rules, parsing, normalization

Examples of justified invariant tests:
- Pipeline state transitions
- Confidence calculation (reinforce/contradict)
- Entity extraction parsing
- Memory type classification logic
"""
