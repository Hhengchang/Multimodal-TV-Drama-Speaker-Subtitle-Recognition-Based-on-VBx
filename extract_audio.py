import librosa
import soundfile as sf
from pathlib import Path
import json


def proccess_rttm(file_lab):
    time_split=[]
    
    with open(file_lab, 'r') as f:
        lines = f.readlines()

    for line in lines:
        start = line.split()[3]
        end = line.split()[5]
        time_split.append((float(start), float(end)))
    
    return time_split
    
def split_audio(input_file, start_time, output_file, split):
    
    sr = 16000
    # 保存切分后的音频
    Path(output_file).mkdir(parents=True, exist_ok=True)

    # 加载音频文件，sr=None 保持原有采样率
    audio, _ = librosa.load(input_file, sr=sr)
    # 计算起始和结束样本点
    start_sample = int(start_time * sr)
    end_sample = len(audio)
    audio_segment = audio[start_sample:end_sample]
    for start,end in split:
        if start == end:
            continue
        start_sr = (start)*sr
        end_sr = (end)*sr
        start_sr = int(start_sr)
        end_sr = int(end_sr)
        audio_extract = audio_segment[start_sr:end_sr]
        sf.write(output_file + f'{start:.3f}_{end:.3f}' + '.wav', audio_extract, sr)

# 使用示例
if __name__ == "__main__":
    # 切分音频，切分开始和结束时间（单位：秒）
    audio_path_list = []
    audio_output_list = []
    lab_file_list = []
    audio_path_list = [f"raw_audio/EP{i:02}.wav" for i in range(1,15+1)]# 输入音频文件名称
    audio_output_list = [f"VBx-master/vbx_audio/EP{i:02}/" for i in range(1,15+1)]# 输出切分后音频文件名称
    lab_file_list = [f'VBx-master/rttm-txt/rttm-txt-{i:02}.txt' for i in range(1,15+1)]
    
    for iii in range(0, 15):
        list_split = proccess_rttm(lab_file_list[iii])
        split_audio(audio_path_list[iii], 0, audio_output_list[iii], list_split)

        print(f'音频切分完成，结果保存在 {audio_output_list[iii]}')
    print("----------- 结果保存在: VBx-master/vbx_audio/ -----------")

