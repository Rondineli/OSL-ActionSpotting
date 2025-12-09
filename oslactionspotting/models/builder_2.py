class DotDict(dict):
    """Dict that supports attribute access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def build_model(cfg):
    """
    cfg is normally a pure dict from mmengine.
    We convert it to a DotDict to allow cfg.type, cfg.backbone, ...
    """
    cfg = DotDict(cfg)

    model_type = cfg.type

    if model_type == "LearnablePooling":
        from .learnablepooling import LearnablePoolingModel
        #from .learnablepooling import LiteLearnablePoolingModel
        backbone = DotDict(cfg.backbone)
        neck = DotDict(cfg.neck)
        head = DotDict(cfg.head)
        post_proc = DotDict(cfg.post_proc) if "post_proc" in cfg else None

        return LearnablePoolingModel(
            backbone=backbone,
            neck=neck,
            head=head,
            post_proc=post_proc
        )
    elif model_type == "ContextAware":
        from .contextaware import ContextAwareModel
        print(cfg)

        backbone = DotDict(cfg.backbone)
        neck = DotDict(cfg.neck)
        head = DotDict(cfg.head)

        return ContextAwareModel(
            #cfg=cfg,
            backbone=backbone,
            head=head,
            neck=neck,
            #runner=None,
        )

    raise ValueError(f"Unknown model type: {model_type}")

