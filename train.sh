#!/bin/bash

set -x

CONFIG_MODEL="${1:-configs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model.py}"

python tools/train.py "$CONFIG_MODEL" --cfg-options \
	dataset.train.data_root=/workspace/datasets/amateur-dataset/ \
	dataset.valid.data_root=/workspace/datasets/amateur-dataset/ \
	dataset.train.path=/datasets/amateur/train_amateur_annotations.json \
	dataset.valid.path=/datasets/amateur/valid_amateur_annotations.json \
	training.max_epochs=200 \
	dataset.test.data_root=/workspace/datasets/amateur-dataset/ \
	dataset.test.path=/datasets/amateur/test_amateur_annotations.json



python tools/infer.py "$CONFIG_MODEL"     \
     --cfg-options dataset.test.data_root=/workspace/datasets/amateur-dataset/ \
     dataset.test.path=/test_annotations.json
     
python tools/evaluate.py "$CONFIG_MODEL" \
    --cfg-options dataset.test.data_root=/workspace/datasets/amateur-dataset/ \
    dataset.test.path=/test_annotations.json dataset.test.metric=loose >> "/tmp/$CONFIG_MODEL_result_1.txt"
