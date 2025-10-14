#!/usr/bin/env bash


# 1.将视频转化为采样率16k、单通道的 wav音频
bash ./video-audio-convert.sh

# 2.处理OCR截取出的 json文件，转成lab文件供VBx聚类使用
python json-lab.py

# 3.将每一集按照lab文件使用VBx聚类，得到rttm文件(在exp/文件夹下)
bash ./VBx-master/run_example.sh

# 4.加入一列终止时间（根据开始时间和持续时间计算得出的）,得到rttm-txt-xx.txt（在rttm-txt/文件夹下）
python rttm_proccess.py

# 5.根据rttm-txt-xx.txt文件中的起始和终止时间切分音频（用于聚类后的数据与标签数据对比）
python extract_audio.py

# 6.提取真实标签的平均embedding和聚类后每一句话的embedding。
# 计算聚类后的每一句话embedding和真实标签的embedding做余弦相似度打分，给虚拟标签分配真实标签
bash ./VBx-master/run_example1.sh

# 7.对比字幕和处理embedding后的最终结果（真实标签）
python output_result.py
