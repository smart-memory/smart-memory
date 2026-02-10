"""
Error harness for SmartMemory tests.

Provides table-driven test infrastructure for common error pathways,
reducing test maintenance burden by consolidating error handling tests.

Per testing philosophy:
- validation errors (400)
- not found (404)
- permission denied (403)
- conflict/idempotency (409)
- dependency failure (502/503)
- timeout/retry (504)
"""
