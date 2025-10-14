#!/bin/bash

mkdir -p raw_audio

# 遍历并转换所有 .mp4 文件
for file in video/*.mp4; do
    output_file="raw_audio/$(basename "$file" .mp4).wav"
    ffmpeg -i "$file" -acodec pcm_s16le -ar 16000 -ac 1 "$output_file"
done

echo "----------- 结果保存在: ./raw_audio/ -----------"
