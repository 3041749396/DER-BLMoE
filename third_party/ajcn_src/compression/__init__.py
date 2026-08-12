try:
    from third_party.ajcn_src.compression.pruning import compute_channel_importance, prune_model, get_optimal_prune_ratios
except Exception:
    compute_channel_importance = None
    prune_model = None
    get_optimal_prune_ratios = None

from third_party.ajcn_src.compression.depthwise_conversion import convert_to_depthwise_separable, convert_layer_weights, convert_block_to_depthwise

__all__ = [
    'compute_channel_importance',
    'prune_model',
    'get_optimal_prune_ratios',
    'convert_to_depthwise_separable',
    'convert_layer_weights',
    'convert_block_to_depthwise',
]
