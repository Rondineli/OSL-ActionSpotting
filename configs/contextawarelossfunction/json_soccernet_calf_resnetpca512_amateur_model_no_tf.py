_base_ = [
    "../_base_/datasets/json/features_clips_CALF.py",  # dataset config
    "../_base_/models/contextawarelossfunction.py",  # model config
    "../_base_/schedules/calf_1000_adam.py" # trainer config
]

work_dir = "outputs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model_model_no_tf"
classes = ["Goal", "Kick-off"]
data_root = "/workspace/datasets/amateur-dataset/"

dataset = dict(
    train=dict(
        path=[
            "/workspace/datasets/amateur-dataset/train/annotations.json"
        ],
        data_root=["/workspace/datasets/amateur-dataset/"],
        classes = ["Goal", "Kick-off"],
        evaluation_frequency=20,
    ),
    valid=dict(
        path=["/workspace/datasets/amateur-dataset/valid/annotations.json"],
        data_root=["/workspace/datasets/amateur-dataset/"],
        classes = ["Goal", "Kick-off"]
    ),
    test=dict(
        path=["/workspace/datasets/amateur-dataset/test/annotations.json"],
        data_root=["/workspace/datasets/amateur-dataset/"],
        classes = ["Goal", "Kick-off"],
        metric='loose'
    ),
)

log_level = "INFO"  # The level of logging

runner = dict(type="runner_JSON")

optimizer = dict(lr=1e-4)
scheduler = dict(type="ReduceLROnPlateau", patience=10)

evaluation_frequency = 20

training = dict(
    criterion = dict(
        type='Combined2x',
        w_1=1.0,
        loss_1=dict(
            type='ContextAwareLoss',
            K=[[-100, -100], [-50, -50], [50, 50], [100, 100]],
            framerate=2,
            pos_radius=8, # changed from 4
            neg_radius=20, # changed from 9
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

