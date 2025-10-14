# Multimodal-TV-Drama-Speaker-Subtitle-Recognition-Based-on-VBx
For the precise identification of speakers in films and TV series.At the same time, use OCR to extract each line of dialogue.Utilizing multimodal fusion to achieve precise recognition of the speakers' subtitles in films and TV series.

# Usage
## Installation
```
git clone https://github.com/Hhengchang/Multimodal-TV-Drama-Speaker-Subtitle-Recognition-Based-on-VBx.git

# 1.可以使用pip进行安装env
conda create -n VBx python=3.9
conda activate VBx
pip install -r requirements.txt

# 2.也可以使用conda进行安装
conda env create -f environment.yml #会直接得到一个名为VBx的虚拟环境同时安装所有依赖
conda activate VBx
```
## Change Pre-train Models
在 **./VBx-master/VBx/models/ResNet101_16kHz** 路径下更换提取speaker embedding的预训练模型，其他预训练模型可在[pretrain_model](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md) 下载
## file-path
- json：存放通过ocr处理得到的字幕、演员表JSON文件（包含时间戳）
- data：存放带少量标签的真实说话人音频（0/ ,1/, 2/ ....每个子文件夹下存放一个说话人音频）
- video：存放原始视频
- raw_audio：存放经过ffmpeg处理后的音频（将视频转为音频）
- results：保存最终的json文件
- truth_speaker_embedding：存储保存的演员声纹库（每个演员的embedding）
- VBx-master：VBx模型代码
## Prepare Dataset
- 对file-path路径下的 **json、data、video** 中的文件替换为想要提取电视剧的数据文件
- 更改actor_list_mapping.txt文件，按照data文件夹中的真实说话人音频（0/ ,1/, 2/）和真实演员名称对应
```
# 如：开端电视剧
0 李诗情			
1 肖鹤云			
2 张成			
3 杜劲松			
4 王兴德 			
5 陶映红			
6 钥匙男			
7 卢笛	
```
## inference
将根目录改为当前目录,执行下面命令即可得到最终结果，保存在results文件夹中
```
bash run.sh
```
## Reference
[VBx](https://github.com/BUTSpeechFIT/VBx)







