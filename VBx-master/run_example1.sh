#!/usr/bin/env bash




mkdir -p ./truth_speaker_embedding

# 提取带标签数据的平均 embedding
for speakers in $(ls ./data/)
do
    python VBx-master/VBx/extract_embedding.py \
        --gpus '0' \
        --model ResNet101\
        --weights VBx-master/VBx/models/ResNet101_16kHz/nnet/raw_81.pth \
        --backend pytorch\
        --mode speaker \
        --speaker-dir ./data/${speakers}\
        --output-file ./truth_speaker_embedding/speaker${speakers}_embedding.npy

done

echo "----------- 结果保存在: ./truth_speaker_embedding/ -----------"

# 对比标签 embedding和聚类 embedding
python VBx-master/VBx/extract_embedding.py \
        --gpus '0' \
        --model ResNet101\
        --weights VBx-master/VBx/models/ResNet101_16kHz/nnet/raw_81.pth \
        --backend pytorch\
        --single_read_embedding True


rm -rf VBx-master/VBx/models/ResNet101_16kHz/nnet/raw_81.pth


