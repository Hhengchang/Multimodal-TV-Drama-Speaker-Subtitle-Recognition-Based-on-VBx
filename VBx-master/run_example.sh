#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p VBx-master/exp
cat VBx-master/VBx/models/ResNet101_16kHz/nnet/raw_81.pth.zip.part* > VBx-master/VBx/models/ResNet101_16kHz/nnet/raw_81.pth.zip
unzip VBx-master/VBx/models/ResNet101_16kHz/nnet/raw_81.pth.zip -d VBx-master/VBx/models/ResNet101_16kHz/nnet/
rm -rf VBx-master/VBx/models/ResNet101_16kHz/nnet/raw_81.pth.zip

for audio in $(ls ./raw_audio)
do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > VBx-master/exp/list.txt

      # run feature and x-vectors extraction
      python VBx-master/VBx/predict.py \
          --gpus '0' \
          --model ResNet101\
          --in-file-list VBx-master/exp/list.txt \
          --in-lab-dir VBx-master/vad \
          --in-wav-dir ./raw_audio \
          --out-ark-fn VBx-master/exp/${filename}.ark \
          --out-seg-fn VBx-master/exp/${filename}.seg \
          --weights VBx-master/VBx/models/ResNet101_16kHz/nnet/raw_81.pth \
          --backend pytorch

done


# 贝叶斯自动优化参数
python VBx-master/VBx/bys.py \
    --mode optimize-silhouette \
    --visualization-dir ./optimization_plots \
    --xvec-ark-file VBx-master/exp/EP01.ark \
    --segments-file VBx-master/exp/EP01.seg \
    --xvec-transform VBx-master/VBx/models/ResNet101_16kHz/transform.h5 \
    --plda-file VBx-master/VBx/models/ResNet101_16kHz/plda \
    --threshold 0.01 \
    --lda-dim 128 \
    --optimization-iterations 60


JSON_FILE="./optimization_plots/optimization_summary.json"

# 检查JSON文件是否存在
if [ ! -f "$JSON_FILE" ]; then
    echo "错误: JSON文件 $JSON_FILE 不存在"
    exit 1
fi

# 使用Python解析JSON文件并格式化为3位小数
read_values() {
    python3 - <<END
import json
import sys

try:
    with open('$JSON_FILE', 'r') as f:
        data = json.load(f)
    
    # 提取参数并格式化为3位小数
    fa = round(data['best_params']['Fa'], 3)
    fb = round(data['best_params']['Fb'], 3)
    loopP = round(data['best_params']['loopP'], 3)
    best_score = round(data['best_score'], 6)  # 分数可以保留更多位数
    
    print(f"{fa:.3f} {fb:.3f} {loopP:.3f} {best_score:.6f}")
    
except Exception as e:
    print(f"错误: {e}")
    sys.exit(1)
END
}

# 读取值
result=$(read_values)

# 检查是否成功读取
if [ $? -ne 0 ]; then
    echo "解析JSON文件时出错"
    exit 1
fi

# 分割结果到各个变量
Fa=$(echo $result | awk '{print $1}')
Fb=$(echo $result | awk '{print $2}')
loopP=$(echo $result | awk '{print $3}')
best_score=$(echo $result | awk '{print $4}')

# 输出结果
echo "最佳轮廓系数: $best_score"
echo "Fa = $Fa"
echo "Fb = $Fb"
echo "loopP = $loopP"



for audio in $(ls ./raw_audio)
do
    #   run variational bayes on top of x-vectors
      python VBx-master/VBx/vbhmm.py \
        --init AHC+VB \
        --out-rttm-dir VBx-master/exp \
        --xvec-ark-file VBx-master/exp/${filename}.ark \
        --segments-file VBx-master/exp/${filename}.seg \
        --xvec-transform VBx-master/VBx/models/ResNet101_16kHz/transform.h5 \
        --plda-file VBx-master/VBx/models/ResNet101_16kHz/plda \
        --threshold -0.01 \
        --lda-dim 128 \
        --Fa $Fa \
        --Fb $Fb \
        --loopP $loopP

        
        # --threshold -0.01 \
        # --lda-dim 128 \
        # --Fa 0.4 \
        # --Fb 5.4 \
        # --loopP 0.999 
done





