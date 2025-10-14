# Multimodal-TV-Drama-Speaker-Subtitle-Recognition-Based-on-VBx
For the precise identification of speakers in films and TV series.At the same time, use OCR to extract each line of dialogue.Utilizing multimodal fusion to achieve precise recognition of the speakers' subtitles in films and TV series.

# Usage
## Installation
```
git clone https://github.com/Hhengchang/Multimodal-TV-Drama-Speaker-Subtitle-Recognition-Based-on-VBx.git

# 1.Installed using pip.
conda create -n VBx python=3.9
conda activate VBx
pip install -r requirements.txt

# 2.Installed using conda.
conda env create -f environment.yml #You will directly obtain a virtual environment named VBx and all the dependencies will be installed simultaneously.
conda activate VBx
```
## Change Pre-train Models
The pre-trained model for extracting speaker embeddings to the model located at **./VBx-master/VBx/models/ResNet101_16kHz**. Changing pre-trained models can be downloaded from [pretrain_model](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md) 
## file-path
- json: JSON file storing subtitles and cast information obtained through OCR processing (including timestamps)
- data: Contains real speaker audio with a few labels (each subfolder stores one speaker's audio)
- video: Stores the original video
- raw_audio: Stores the audio processed by ffmpeg (converting video to audio)
- results: Saves the final JSON file
- truth_speaker_embedding: Stores the saved actor voiceprint database (embedding of each actor)
- VBx-master: Code of the VBx model
## Prepare Dataset
- Replace the files in the "file-path" directory (such as "json", "data", "video") with the files containing the data for the TV series that you want to extract.
- Modify the "actor_list_mapping.txt" file, ensuring that it corresponds to the actual speaker audio (0/, 1/, 2/) and the actual actor names in the "data" folder.
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
Change the root directory to the current directory. Then, execute the following command to obtain the final result, which will be saved in the "results" folder.
```
bash run.sh
```
## Reference
[VBx](https://github.com/BUTSpeechFIT/VBx)
[VBx-training](https://github.com/phonexiaresearch/VBx-training-recipe)












