"""Makes tests/ a package, and — critically — points ZOVIVE_HOME at a
throwaway temp directory *before* any test module imports `paths`, so
the test suite never touches a real deployment's var/ directory (db,
snapshots, logs). This runs once, at test collection time, because
Python only executes a package's __init__.py the first time anything
under it is imported.
"""

import os
import tempfile

os.environ.setdefault("ZOVIVE_HOME", tempfile.mkdtemp(prefix="zovive_test_"))
