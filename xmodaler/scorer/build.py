from xmodaler.utils.registry import Registry

SCORER_REGISTRY = Registry("SCORER")
SCORER_REGISTRY.__doc__ = """
Registry for scorer
"""

def build_scorer(cfg):
    scorer = SCORER_REGISTRY.get(cfg.SCORER.NAME)(cfg)
    return scorer