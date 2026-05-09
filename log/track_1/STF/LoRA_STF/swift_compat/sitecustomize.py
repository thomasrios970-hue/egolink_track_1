try:
    import transformers.integrations.tensor_parallel as tp
    if not hasattr(tp, 'EmbeddingParallel') and hasattr(tp, 'ColwiseParallel'):
        tp.EmbeddingParallel = tp.ColwiseParallel
except Exception:
    pass
