"""Fashion retail modelling package."""
import os
from pathlib import Path
import tempfile

# A writable cache also supports restricted lab/container accounts.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fashion_retail_matplotlib"))
