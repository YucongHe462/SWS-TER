# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

from mmcv.utils import collect_env as collect_basic_env
from mmcv.utils import get_git_hash

import mmrotate


def collect_env():
    """Collect environment information."""
    try:
        env_info = collect_basic_env()
    except UnicodeDecodeError:
        # Windows shells may return compiler metadata in an encoding that
        # mmcv 1.x cannot decode with the active locale. Environment logging
        # should not prevent training from starting.
        env_info = {}
    repo_root = Path(__file__).resolve().parents[2]
    revision = ('+' + get_git_hash(digits=7)
                if (repo_root / '.git').is_dir() else '')
    env_info['MMRotate'] = mmrotate.__version__ + revision
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
