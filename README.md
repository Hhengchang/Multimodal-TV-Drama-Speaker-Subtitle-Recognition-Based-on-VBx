# Multimodal-TV-Drama-Speaker-Subtitle-Recognition-Based-on-VBx
For the precise identification of speakers in films and TV series.At the same time, use OCR to extract each line of dialogue.Utilizing multimodal fusion to achieve precise recognition of the speakers' subtitles in films and TV series.

# Usage
## Installation
```
git clone https://github.com/Hhengchang/Multimodal-TV-Drama-Speaker-Subtitle-Recognition-Based-on-VBx

# 1.可以使用pip进行安装env
conda create -n VBx python=3.9
conda activate VBx
pip install -r requirements.txt
# 2.也可以使用conda进行安装
conda env create -f environment.yml #会直接得到一个名为VBx的虚拟环境同时安装所有依赖
conda activate VBx
```
# 文件夹对应存储文件：

- huawei/json文件夹下存放通过ocr处理得到的字幕、演员表JSON文件（包含时间戳）
- huawei/data文件夹下存放带少量标签的真实说话人音频（0/ ,1/, 2/ ....每个子文件夹下存放一个说话人音频）
- huawei/video文件夹下存放原始视频
- huawei/raw_audio文件夹下存放经过ffmpeg处理后的音频（将视频转为音频）
- huawei/results文件夹保存最终赛题要求的json文件
- huawei/truth_speaker_embedding文件夹下存储保存的演员声纹库（每个演员的embedding）

- huawei/VBx-master文件夹下是模型代码



