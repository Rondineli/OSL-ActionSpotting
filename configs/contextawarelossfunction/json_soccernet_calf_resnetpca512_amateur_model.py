_base_ = [
    #"../_base_/datasets/json/features_clips_CALF.py",  # dataset config
    #"../_base_/models/contextawarelossfunction.py",  # model config
    "../_base_/schedules/calf_1000_adam.py" # trainer config
]

work_dir = "outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model"
cut_classes = ["Goal", "Kick-off"]
classes=cut_classes
#classes = [
#    "Penalty",
#    "Kick-off",
#    "Goal"
#    "Substitution",
#    "Offside",
#    "Shots on target",
#    "Shots off target",
#    "Clearance",
#    "Ball out of play",
#    "Throw-in",
#    "Foul",
#    "Indirect free-kick",
#    "Direct free-kick",
#    "Corner",
#    "Yellow card",
#    "Red card",
#    "Yellow->red card"
#]

data_root = "/workspace/datasets/amateur-dataset/"

dataset = dict(
    input_fps=25,
    extract_fps=2,
    train=dict(
        type="FeatureClipChunksfromJson",
        path="/workspace/datasets/amateur-dataset/train/annotations.json",
        data_root="/workspace/datasets/amateur-dataset/", #"/home/ybenzakour/datasets/SoccerNet/",
        framerate=1,
        chunk_size=120,
        receptive_field=40,
        chunks_per_epoch=700, # prior 6000
        classes=cut_classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        ),
    ),
    valid=dict(
        type="FeatureClipChunksfromJson",
        path="/workspace/datasets/amateur-dataset/valid/annotations.json",
        data_root="/workspace/datasets/amateur-dataset/",
        framerate=1,
        chunk_size=120,
        receptive_field=40,
        chunks_per_epoch=700, # prior 6000
        classes=cut_classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        ),
    ),
    test=dict(
        type="FeatureVideosChunksfromJson",
        path="/workspace/datasets/amateur-dataset/test/annotations.json",
        data_root="/workspace/datasets/amateur-dataset/",
        framerate=1,
        chunk_size=120,
        receptive_field=40,
        chunks_per_epoch=700, # prior 6000
        classes=cut_classes,
        metric="loose",
        results="results_spotting_test",
        dataloader=dict(
            num_workers=1,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        ),
    ),
)

model = dict(
    type='ContextAware',
    #load_weights=None,
    #load_from="outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512/model.pth.tar",
    load_weights="outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512/model.pth.tar",
    reset_head=False,
    reset_neck=False,
    freeze_backbone=False,
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
    pos_radius=5,       # frames; default is often too small (1)
    neg_radius=11,       # frames; increases tolerance to temporal jitter
    lambda_reg=0.5,     # regularization to reduce overfitting
    lambda_neg=0.25,    # reduce penalty of negatives (important for rare events)
    lambda_pos=2.0,     # emphasize positives (especially rare events)
    normalize=True,
)

#runner = dict(
#    type="runner_CALF"
#)


log_level = "INFO"  # The level of logging


#optimizer = dict(lr=1e-4)
optimizer = dict(type="Adam", lr=1e-4)
scheduler = dict(type="ReduceLROnPlateau", patience=10)

evaluation_frequency = 20

training = dict(
    criterion = dict(
        type='Combined2x',
        w_1=0.000367,
        #w_1=1.0,
        loss_1=dict(
            type='ContextAwareLoss',
            K=[[-100, -100], [-50, -50], [50, 50], [100, 100]],
            #K=[[-100, -98, -20, -40, -96, -5, -8, -93, -99, -31, -75, -10, -97, -75, -20, -84, -18],
            #[-50, -49, -10, -20, -48, -3, -4, -46, -50, -15, -37, -5, -49, -38, -10, -42, -9],
            #[50, 49, 60, 10, 48, 3, 4, 46, 50, 15, 37, 5, 49, 38, 10, 42, 9],
            #[100, 98, 90, 20, 96, 5, 8, 93, 99, 31, 75, 10, 97, 75, 20, 84, 18]],
            framerate=1,
            pos_radius=6, # changed from 4
            neg_radius=11, # changed from 9
            hit_radius=0.2,
            miss_radius=0.8,
        ),
        w_2=1.0,
        loss_2=dict(
            type='SpottingLoss',
            lambda_coord=5.0,
            lambda_noobj=0.5,
        )
    )
)

runner = dict(type="runner_JSON")
