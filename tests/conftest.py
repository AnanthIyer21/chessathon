import os
import sys

# Submission files use flat imports because they sit at ZIP root in the
# contest environment. Mirror that here.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))
