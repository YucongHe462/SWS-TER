"""Small runtime fixes for the legacy OpenMMLab toolchain."""

from __future__ import annotations

import os
import inspect
import tempfile
from pathlib import Path


def configure_yapf_cache() -> Path:
    """Keep YAPF grammar caching in a writable, process-safe directory.

    Recent YAPF releases select a user cache directory while MMCV is imported.
    On restricted Windows profiles that lookup can stall. Redirecting only the
    YAPF/platformdirs lookup avoids changing Python's temporary-file behavior.
    """

    root = Path(os.environ.get(
        'SWS_TER_CACHE_DIR', Path(tempfile.gettempdir()) / 'sws-ter-cache'))
    root.mkdir(parents=True, exist_ok=True)

    import platformdirs
    platformdirs.user_cache_dir = lambda *args, **kwargs: str(root)

    # MMCV 1.x passes ``verify=`` to YAPF. YAPF 0.43 removed that keyword.
    from yapf.yapflib import yapf_api
    if 'verify' not in inspect.signature(yapf_api.FormatCode).parameters:
        format_code = yapf_api.FormatCode

        def format_code_compat(*args, verify=None, **kwargs):
            del verify
            return format_code(*args, **kwargs)

        yapf_api.FormatCode = format_code_compat
    return root
