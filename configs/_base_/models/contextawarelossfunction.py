model = dict(
    type='ContextAware',
    load_weights=None,
    backbone=dict(
        type='PreExtactedFeatures',
        encoder='ResNET_TF2_PCA512',
        feature_dim=512,
        output_dim=512,
        framerate=1),
    neck=dict(
        type='CNN++',
        input_size=512, 
        num_classes=2, 
        chunk_size=120, 
        dim_capsule=16,
        receptive_field=40, 
        num_detections=15, 
        framerate=1),
    head=dict(
        type='SpottingCALF',
        num_classes=2,
        dim_capsule=16,
        num_detections=15,
        num_layers=2,
        chunk_size=120),
    # post_proc=dict(
    #     type="NMS",
    #     NMS_window=30,
    #     NMS_threshold=0.0),
)

contextaware_cfg = dict(
    pos_radius=3,       # frames; default is often too small (1)
    neg_radius=9,       # frames; increases tolerance to temporal jitter
    lambda_reg=0.5,     # regularization to reduce overfitting
    lambda_neg=0.25,    # reduce penalty of negatives (important for rare events)
    lambda_pos=2.0,     # emphasize positives (especially rare events)
    normalize=True,
)

runner = dict(
    type="runner_CALF"
)
