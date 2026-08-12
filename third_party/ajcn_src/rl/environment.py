
"""Compatibility CompressionEnv stub.

The uploaded AJCN archive used in this project did not include rl/environment.py,
although cuda/distributed scripts import it. The strict integration path in
models/ajcn_official_adapter.py does not rely on this class; it is provided only
so original AJCN auxiliary scripts remain importable.
"""

class CompressionEnv:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The uploaded AJCN archive lacks the original rl/environment.py. "
            "Use models.ajcn_official_adapter.OfficialAJCNTrainer for the "
            "integrated evaluation path, or replace third_party/AJCN-main with "
            "the complete official repository."
        )
