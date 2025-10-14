#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Authors: Lukas Burget, Federico Landini, Jan Profant
# @Emails: burget@fit.vutbr.cz, landini@fit.vutbr.cz, jan.profant@phonexia.com

import argparse
import logging
import os
import time
import json
from pathlib import Path
import numpy as np
import glob

from tqdm import tqdm
import kaldi_io
import onnxruntime
import soundfile as sf
import torch.backends
from collections import defaultdict
import features
from models.resnet import *

torch.backends.cudnn.enabled = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_line(line):
    parts = line.split()
    virtual_id = parts[8]  # 第9列（索引8）
    true_id = parts[11]    # 第12列（索引11）
    score = float(parts[12])  # 最后一列（索引12）
    return virtual_id, true_id, score

def apply_mapping_to_file(input_file, output_file, mapping,speakers_sum):
    """
    将映射应用到原始文件，生成新的文件
    """
    # 读取原始文件
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 处理每一行
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 9:
            virtual_id = parts[8]  # 第9列是虚拟ID
            
            # 应用映射
            if virtual_id in mapping:
                # 其他人ID(最大标号)
                if virtual_id == speakers_sum:
                    mapped_id=speakers_sum
                else:
                    mapped_id = mapping[virtual_id]
                # 如果映射为None，保留原始值或标记为特殊值
                if mapped_id is None:
                    parts[11] = "NULL"  # 将真实ID标记为NULL
                else:
                    parts[11] = str(mapped_id)
        
        new_lines.append(" ".join(parts) + "\n")
    
    # 写入新文件
    with open(output_file, 'w') as f:
        f.writelines(new_lines)

def main(input, output, speakers_sum):
    # 读取文件
    
    with open(input, 'r') as f:
        lines = f.readlines()

    # 统计每个虚拟ID对应的真实ID的出现次数和总得分
    virtual_to_true = defaultdict(lambda: defaultdict(int))
    virtual_to_total_score = defaultdict(lambda: defaultdict(float))

    for line in lines:
        if line.strip():
            virtual_id, true_id, score = parse_line(line)
            virtual_to_true[virtual_id][true_id] += 1
            virtual_to_total_score[virtual_id][true_id] += score

    # 为每个虚拟ID选择出现次数最多的真实ID
    # 如果出现次数相同，选择总得分更高的真实ID
    mapping = {}
    for virtual_id, true_counts in virtual_to_true.items():
        max_count = max(true_counts.values())
        # 找出所有出现次数等于max_count的真实ID
        candidates = [true_id for true_id, count in true_counts.items() if count == max_count]
        if len(candidates) == 1:
            mapping[virtual_id] = candidates[0]
        else:
            # 选择总得分最高的真实ID
            best_true_id = None
            best_score = -1
            for true_id in candidates:
                total_score = virtual_to_total_score[virtual_id][true_id]
                if total_score > best_score:
                    best_score = total_score
                    best_true_id = true_id
            mapping[virtual_id] = best_true_id

    apply_mapping_to_file(input, output, mapping, speakers_sum)
    # 输出映射结果
    print("虚拟说话人ID -> 真实说话人ID 映射:")
    for virtual_id, true_id in sorted(mapping.items(), key=lambda x: int(x[0])):
        print(f"虚拟ID {virtual_id} -> 真实ID {true_id}")

    # 计算准确率
    total_lines = 0
    correct_lines = 0
    for line in lines:
        if line.strip():
            virtual_id, true_id, _ = parse_line(line)
            if mapping.get(virtual_id) == true_id:
                correct_lines += 1
            total_lines += 1

    accuracy = correct_lines / total_lines * 100
    # print(f"\n准确率: {accuracy:.2f}%")
    
    return accuracy



def actor_list_mapping(input, output, actor_list):
    # 读取原始文件
    with open(input, 'r') as f:
        lines = f.readlines()
    
    actor_mapping={}
    # 读取演员表
    with open(actor_list, 'r') as f1:
        actor_lines = f1.readlines()
        actor_mapping = {actor_line.strip().split()[0]:actor_line.strip().split()[1] for actor_line in actor_lines}
        
     # 处理每一行
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 12:
            truth_id = parts[11]  # 第12列是真实ID
            
            # 应用映射
            if truth_id in actor_mapping:
                mapped_id = actor_mapping[truth_id]
                
                # 如果映射为None，保留原始值或标记为特殊值
                if mapped_id is None:
                    parts[11] = "NULL"  # 将真实ID标记为NULL
                else:
                    parts[11] = str(mapped_id)
            else:
                mapped_id = '其他'
                parts[11] = str(mapped_id)
                
        new_lines.append(" ".join(parts) + "\n")
        
    # 写入新文件
    with open(output, 'w') as f:
        f.writelines(new_lines)

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        if self.name:
            logger.info(f'Start: {self.name}: ')

    def __exit__(self, type, value, traceback):
        if self.name:
            logger.info(f'End:   {self.name}: Elapsed: {time.time() - self.tstart} seconds')
        else:
            logger.info(f'End:   {self.name}: ')


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1: 第一个向量 (numpy array)
        vec2: 第二个向量 (numpy array)
    
    Returns:
        float: 余弦相似度，范围[-1, 1]
    """
    # 确保向量形状一致
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    
    # 计算模长
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # 避免除以零
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
    
    return similarity


def initialize_gpus(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def load_utt(ark, utt, position):
    with open(ark, 'rb') as f:
        f.seek(position - len(utt) - 1)
        ark_key = kaldi_io.read_key(f)
        assert ark_key == utt, f'Keys does not match: `{ark_key}` and `{utt}`.'
        mat = kaldi_io.read_mat(f)
        return mat


def write_txt_vectors(path, data_dict):
    """ Write vectors file in text format.

    Args:
        path (str): path to txt file
        data_dict: (Dict[np.array]): name to array mapping
    """
    with open(path, 'w') as f:
        for name in sorted(data_dict):
            f.write(f'{name}  [ {" ".join(str(x) for x in data_dict[name])} ]{os.linesep}')


def get_embedding(fea, model, label_name=None, input_name=None, backend='pytorch'):
    if backend == 'pytorch':
        data = torch.from_numpy(fea).to(device)
        data = data[None, :, :]
        data = torch.transpose(data, 1, 2)
        spk_embeds = model(data)
        return spk_embeds.data.cpu().numpy()[0]
    elif backend == 'onnx':
        return model.run([label_name],
                         {input_name: fea.astype(np.float32).transpose()
                         [np.newaxis, :, :]})[0].squeeze()


class ReferenceVectorExporter:
    """参考说话人向量导出器"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.reference_vectors = {}
        os.makedirs(output_dir, exist_ok=True)
    
    def add_reference_speaker(self, speaker_name, xvector):
        """添加参考说话人向量"""
        if speaker_name not in self.reference_vectors:
            self.reference_vectors[speaker_name] = []
        self.reference_vectors[speaker_name].append(xvector)
        logger.info(f"Added reference vector for {speaker_name}")
    
    def export_reference_vectors(self, format='all'):
        """导出参考向量到文件"""
        if not self.reference_vectors:
            logger.warning("No reference vectors to export")
            return
        
        # 计算每个说话人的平均向量
        averaged_vectors = {}
        for speaker_name, vectors in self.reference_vectors.items():
            averaged_vectors[speaker_name] = np.mean(vectors, axis=0)
            logger.info(f"Speaker {speaker_name}: {len(vectors)} vectors averaged")
        
        # 导出不同格式
        if format in ['all', 'ark']:
            self._export_kaldi_ark(averaged_vectors)
        if format in ['all', 'npy']:
            self._export_numpy(averaged_vectors)
        if format in ['all', 'json']:
            self._export_json(averaged_vectors)
        if format in ['all', 'txt']:
            self._export_text(averaged_vectors)
    
    def _export_kaldi_ark(self, vectors_dict):
        """导出为Kaldi ark格式"""
        ark_file = os.path.join(self.output_dir, 'reference_vectors.ark')
        with open(ark_file, 'wb') as f:
            for speaker_name, vector in vectors_dict.items():
                kaldi_io.write_vec_flt(f, vector, key=speaker_name)
        logger.info(f"Reference vectors exported to Kaldi ark: {ark_file}")
    
    def _export_numpy(self, vectors_dict):
        """导出为NumPy格式"""
        npy_file = os.path.join(self.output_dir, 'reference_vectors.npy')
        data = {
            'vectors': list(vectors_dict.values()),
            'names': list(vectors_dict.keys())
        }
        np.save(npy_file, data)
        logger.info(f"Reference vectors exported to NumPy: {npy_file}")
    
    def _export_json(self, vectors_dict):
        """导出为JSON格式"""
        json_file = os.path.join(self.output_dir, 'reference_vectors.json')
        data = {
            'vectors': [vector.tolist() for vector in vectors_dict.values()],
            'names': list(vectors_dict.keys())
        }
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Reference vectors exported to JSON: {json_file}")
    
    def _export_text(self, vectors_dict):
        """导出为文本格式"""
        txt_file = os.path.join(self.output_dir, 'reference_vectors.txt')
        with open(txt_file, 'w') as f:
            for speaker_name, vector in vectors_dict.items():
                f.write(f"{speaker_name} [{' '.join(map(str, vector))}]\n")
        logger.info(f"Reference vectors exported to text: {txt_file}")


def find_audio_files(data_root_dir, extensions=['.wav', '.flac', '.mp3']):
    """查找数据根目录下所有说话人文件夹中的音频文件
    
    Args:
        data_root_dir: 数据根目录路径，包含多个说话人子文件夹
        extensions: 支持的音频文件扩展名列表
        
    Returns:
        audio_files: 所有音频文件的完整路径列表
        speaker_map: 文件名到说话人名称的映射字典
    """
    audio_files = []
    speaker_map = {}
    
    # 检查输入目录是否存在
    if not os.path.exists(data_root_dir):
        raise ValueError(f"Data directory {data_root_dir} does not exist")
    
    logger.info(f"Scanning directory: {data_root_dir}")
    
    # 获取所有子文件夹（说话人文件夹）
    speaker_dirs = []
    for item in os.listdir(data_root_dir):
        item_path = os.path.join(data_root_dir, item)
        if os.path.isdir(item_path):
            speaker_dirs.append(item_path)
            logger.info(f"Found speaker directory: {item}")
    
    if not speaker_dirs:
        raise ValueError(f"No speaker directories found in {data_root_dir}")
    
    logger.info(f"Found {len(speaker_dirs)} speaker directories")
    
    # 处理每个说话人文件夹
    for speaker_dir in speaker_dirs:
        speaker_name = os.path.basename(speaker_dir)
        logger.info(f"Processing speaker: {speaker_name}")
        
        audio_count = 0
        # 查找该说话人文件夹中的所有音频文件
        for ext in extensions:
            pattern = os.path.join(speaker_dir, f'*{ext}')
            files = glob.glob(pattern)
            for audio_file in files:
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                audio_files.append(audio_file)
                speaker_map[base_name] = speaker_name
                audio_count += 1
                logger.debug(f"  Found audio: {base_name}")
        
        logger.info(f"  Found {audio_count} audio files for speaker {speaker_name}")
    
    return audio_files, speaker_map


def generate_vad_labels(audio_files, vad_dir, min_duration=0.1, max_duration=10.0):
    """为音频文件生成简单的VAD标签（全段有语音）"""
    os.makedirs(vad_dir, exist_ok=True)
    
    for audio_file in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        vad_file = os.path.join(vad_dir, f'{base_name}.lab')
        
        # 读取音频获取时长
        try:
            signal, samplerate = sf.read(audio_file)
            duration = len(signal) / samplerate
            
            # 如果音频太长，分割成多个段
            if duration > max_duration:
                segments = []
                start = 0.0
                while start < duration:
                    end = min(start + max_duration, duration)
                    if end - start >= min_duration:
                        segments.append((start, end))
                    start = end
            else:
                segments = [(0.0, duration)] if duration >= min_duration else []
            
            # 写入VAD标签文件
            with open(vad_file, 'w') as f:
                for start, end in segments:
                    f.write(f"{start:.3f} {end:.3f}\n")
            
            logger.info(f"Generated VAD for {base_name}: {len(segments)} segments")
            
        except Exception as e:
            logger.warning(f"Failed to process {audio_file}: {e}")


def extract_single_embedding(audio_file, model, samplerate, backend='pytorch', 
                           label_name=None, input_name=None, vad_file=None):
    """提取单段音频的embedding"""
    try:
        # 读取音频
        signal, file_samplerate = sf.read(audio_file)
        if file_samplerate != samplerate:
            logger.warning(f"Sample rate mismatch: expected {samplerate}, got {file_samplerate}")
            return None
        
        # 读取或生成VAD标签
        if vad_file and os.path.exists(vad_file):
            labs = np.atleast_2d(np.loadtxt(vad_file, usecols=(0, 1)) * samplerate).astype(int)
        else:
            # 如果没有VAD文件，使用整个音频
            duration = len(signal) / samplerate
            labs = np.array([[0, int(duration * samplerate)]])
        
        # 特征提取参数设置
        if samplerate == 8000:
            noverlap = 120
            winlen = 200
            window = features.povey_window(winlen)
            fbank_mx = features.mel_fbank_mx(
                winlen, samplerate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=3700, htk_bug=False)
        elif samplerate == 16000:
            noverlap = 240
            winlen = 400
            window = features.povey_window(winlen)
            fbank_mx = features.mel_fbank_mx(
                winlen, samplerate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
        else:
            raise ValueError(f'Only 8kHz and 16kHz are supported. Got {samplerate} instead.')

        LC = 150
        RC = 149

        np.random.seed(3)
        signal = features.add_dither((signal*2**15).astype(int))

        # 处理每个语音段
        xvectors = []
        for segnum in range(len(labs)):
            seg = signal[labs[segnum, 0]:labs[segnum, 1]]
            if seg.shape[0] > 0.01*samplerate:
                seg = np.r_[seg[noverlap // 2 - 1::-1],
                            seg, seg[-1:-winlen // 2 - 1:-1]]
                fea = features.fbank_htk(seg, window, noverlap, fbank_mx,
                                         USEPOWER=True, ZMEANSOURCE=True)
                fea = features.cmvn_floating_kaldi(fea, LC, RC, norm_vars=False).astype(np.float32)

                slen = len(fea)
                # if slen < 144:  # 最小帧数要求
                #     continue
                
                # 提取该段的embedding
                data = fea[:144] if slen > 144 else fea  # 取前144帧或全部
                xvector = get_embedding(
                    data, model, label_name=label_name, input_name=input_name, backend=backend)
                
                if not np.isnan(xvector).any():
                    xvectors.append(xvector)
        
        if xvectors:
            return np.mean(xvectors, axis=0)  # 返回该音频的平均embedding
        else:
            return None
            
    except Exception as e:
        logger.error(f"Failed to extract embedding from {audio_file}: {e}")
        return None


def extract_speaker_embeddings(speaker_audio_files, model, samplerate, backend='pytorch', 
                              label_name=None, input_name=None, vad_dir=None):
    """提取一个说话人多段音频的平均embedding"""
    speaker_xvectors = []
    
    for audio_file in speaker_audio_files:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        vad_file = os.path.join(vad_dir, f'{base_name}.lab') if vad_dir else None
        
        xvector = extract_single_embedding(audio_file, model, samplerate, backend, 
                                         label_name, input_name, vad_file)
        
        if xvector is not None:
            speaker_xvectors.append(xvector)
            logger.info(f"Extracted embedding from {base_name}")
    
    if speaker_xvectors:
        return np.mean(speaker_xvectors, axis=0)  # 返回说话人的平均embedding
    else:
        return None

def save_embedding(embedding, name, output_file, format='npy'):
    """保存单个embedding到文件"""
    if format == 'npy':
        np.save(output_file, {name: embedding})
    elif format == 'ark':
        with open(output_file, 'wb') as f:
            kaldi_io.write_vec_flt(f, embedding, key=name)
    elif format == 'json':
        data = {name: embedding.tolist()}
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == 'txt':
        with open(output_file, 'w') as f:
            f.write(f"{name} [{' '.join(map(str, embedding))}]\n")


def save_embeddings_batch(embeddings_dict, output_file, format='npy'):
    """批量保存多个embedding到文件"""
    if format == 'npy':
        np.save(output_file, embeddings_dict)
    elif format == 'ark':
        with open(output_file, 'wb') as f:
            for name, embedding in embeddings_dict.items():
                kaldi_io.write_vec_flt(f, embedding, key=name)
    elif format == 'json':
        data = {name: embedding.tolist() for name, embedding in embeddings_dict.items()}
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == 'txt':
        with open(output_file, 'w') as f:
            for name, embedding in embeddings_dict.items():
                f.write(f"{name} [{' '.join(map(str, embedding))}]\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='', help='use gpus (passed to CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--model', required=False, type=str, default=None, help='name of the model')
    parser.add_argument('--weights', required=True, type=str, default=None, help='path to pretrained model weights')
    parser.add_argument('--model-file', required=False, type=str, default=None, help='path to model file')
    parser.add_argument('--ndim', required=False, type=int, default=64, help='dimensionality of features')
    parser.add_argument('--embed-dim', required=False, type=int, default=256, help='dimensionality of the emb')
    parser.add_argument('--seg-len', required=False, type=int, default=144, help='segment length')
    parser.add_argument('--seg-jump', required=False, type=int, default=24, help='segment jump')
    parser.add_argument('--samplerate', required=False, type=int, default=16000, choices=[8000, 16000],
                       help='audio sample rate')
    
    # 输入模式选择
    parser.add_argument('--mode', required=False, choices=['single', 'speaker', 'batch'],
                       help='extraction mode: single file, speaker average, or batch processing')
    
    # 单文件模式参数
    parser.add_argument('--input-file', required=False, type=str, 
                       help='input audio file (for single file mode)')
    
    # 说话人模式参数
    parser.add_argument('--speaker-dir', required=False, type=str,
                       help='directory with speaker audio files (for speaker mode)')
    
    parser.add_argument('--single_read_embedding', required=False, type=str,
                       help='directory with speaker audio files (for speaker mode)')
    
    # 批量模式参数
    parser.add_argument('--data-root-dir', required=False, type=str,
                       help='root directory with multiple speaker directories (for batch mode)')
    
    parser.add_argument('--vad-dir', required=False, type=str, default='./vad_labels',
                       help='directory for VAD labels')
    
    # 输出参数
    parser.add_argument('--output-file', required=False, type=str, 
                       help='output file for embeddings')
    parser.add_argument('--output-format', required=False, default='npy', 
                       choices=['ark', 'npy', 'json', 'txt'],
                       help='output format for embeddings')
    
    parser.add_argument('--backend', required=False, default='pytorch', choices=['pytorch', 'onnx'],
                       help='backend that is used for x-vector extraction')
    
    # 参考说话人向量导出
    parser.add_argument('--export-reference', required=False, action='store_true',
                       help='Export reference speaker vectors in batch mode')
    parser.add_argument('--reference-output-dir', required=False, type=str, default='./reference_vectors',
                       help='Directory to store reference vectors')
    parser.add_argument('--reference-format', required=False, type=str, default='all',
                       choices=['all', 'ark', 'npy', 'json', 'txt'],
                       help='Format for reference vectors export')
    parser.add_argument('--auto-vad', required=False, action='store_true',
                       help='Automatically generate VAD labels for entire audio')

    args = parser.parse_args()

    device = ''
    if args.gpus != '':
        logger.info(f'Using GPU: {args.gpus}')
        initialize_gpus(args)
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    model, label_name, input_name = '', None, None

    if args.backend == 'pytorch':
        if args.model_file is not None:
            model = torch.load(args.model_file)
            model = model.to(device)
        elif args.model is not None and args.weights is not None:
            model = eval(args.model)(feat_dim=args.ndim, embed_dim=args.embed_dim)
            model = model.to(device)
            checkpoint = torch.load(args.weights, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.eval()
    elif args.backend == 'onnx':
        model = onnxruntime.InferenceSession(args.weights)
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
    else:
        raise ValueError('Wrong combination of --model/--weights/--model_file '
                         'parameters provided (or not provided at all)')

    # 根据模式处理
    if args.mode == 'single':
        # 单文件模式
        if not args.input_file:
            raise ValueError("Input file is required for single file mode")
        
        logger.info(f"Extracting embedding from single file: {args.input_file}")
        embedding = extract_single_embedding(
            args.input_file, model, args.samplerate, args.backend, 
            label_name, input_name, args.vad_dir
        )
        
        if embedding is not None:
            # 保存单个embedding
            base_name = os.path.splitext(os.path.basename(args.input_file))[0]
            save_embedding(embedding, base_name, args.output_file, args.output_format)
            logger.info(f"Embedding saved to {args.output_file}")
        else:
            logger.error("Failed to extract embedding")
            
        logger.info("------------------Processing completed!----------------------")
        
    elif args.mode == 'speaker':
        # 说话人模式
        if not args.speaker_dir:
            raise ValueError("Speaker directory is required for speaker mode")
        
        # 查找说话人目录中的所有音频文件
        audio_files = []
        for ext in ['.wav', '.flac', '.mp3']:
            audio_files.extend(glob.glob(os.path.join(args.speaker_dir, f'*{ext}')))
        
        if not audio_files:
            raise ValueError(f"No audio files found in {args.speaker_dir}")
        
        logger.info(f"Extracting average embedding from {len(audio_files)} files in {args.speaker_dir}")
        embedding = extract_speaker_embeddings(
            audio_files, model, args.samplerate, args.backend,
            label_name, input_name, args.vad_dir
        )
        
        if embedding is not None:
            # 保存说话人平均embedding
            speaker_name = os.path.basename(args.speaker_dir)
            save_embedding(embedding, speaker_name, args.output_file, args.output_format)
            logger.info(f"Speaker embedding saved to {args.output_file}")
        else:
            logger.error("Failed to extract speaker embedding")
            
        logger.info("------------------Processing completed!----------------------")
        
    elif args.mode == 'batch':
        # 批量模式
        if not args.data_root_dir:
            raise ValueError("Data root directory is required for batch mode")
        
        # 查找所有音频文件和说话人映射
        audio_files, speaker_map = find_audio_files(args.data_root_dir)
        
        if args.auto_vad:
            logger.info("Generating automatic VAD labels...")
            generate_vad_labels(audio_files, args.vad_dir)
        
        # 按说话人分组音频文件
        speaker_files = {}
        for audio_file in audio_files:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            speaker_name = speaker_map[base_name]
            if speaker_name not in speaker_files:
                speaker_files[speaker_name] = []
            speaker_files[speaker_name].append(audio_file)
        
        # 初始化参考向量导出器
        reference_exporter = ReferenceVectorExporter(args.reference_output_dir) if args.export_reference else None
        
        # 处理每个说话人
        all_embeddings = {}
        for speaker_name, files in speaker_files.items():
            logger.info(f"Processing speaker: {speaker_name} with {len(files)} files")
            
            embedding = extract_speaker_embeddings(
                files, model, args.samplerate, args.backend,
                label_name, input_name, args.vad_dir
            )
            
            if embedding is not None:
                all_embeddings[speaker_name] = embedding
                if reference_exporter:
                    reference_exporter.add_reference_speaker(speaker_name, embedding)
                logger.info(f"Successfully processed speaker: {speaker_name}")
            else:
                logger.warning(f"Failed to process speaker: {speaker_name}")
        
        # 保存所有embedding
        save_embeddings_batch(all_embeddings, args.output_file, args.output_format)
        
        # 导出参考向量
        if reference_exporter:
            reference_exporter.export_reference_vectors(format=args.reference_format)
    
        logger.info("------------------Processing completed!----------------------")
    
    def single_speaker(input_file):
        # 单文件模式
        if not input_file:
            raise ValueError("Input file is required for single file mode")
        
        embedding = extract_single_embedding(
            input_file, model, args.samplerate, args.backend, 
            label_name, input_name, args.vad_dir
        )
        
        return embedding
    
    def compare_embedding(rttm_txt_list, base, truth_path, out_file, speaker_total_num):
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(rttm_txt_list, 'r') as input_file, open(out_file, 'w') as out_file:
            lines = input_file.readlines()
            
            for line in lines:
                start = line.strip().split()[3]
                start = round(float(start), 3)
                end = line.strip().split()[5]
                end = round(float(end), 3)
                file = f'{base}{start:.3f}_{end:.3f}.wav'
                embedding = single_speaker(file)
                truth_path = Path(truth_path)
                list_one=[]
                for i in truth_path.rglob('*.npy'):
                    emb = np.load(i, allow_pickle=True).item()
                    # 获取id
                    keys = list(emb.keys())[0]
                    # 获取embedding
                    values = emb[keys]
                    similarity = cosine_similarity(embedding, values)
                    list_one.append((int(keys), similarity))
                max_index = max(range(len(list_one)), key=lambda i: list_one[i][1])
                # 最大的余弦分数
                max_value = list_one[max_index][1]
                # 最大余弦分数对应的真实说话人ID
                max_value_id = list_one[max_index][0]
                
                # 分数小的情况下 设为其他说话人
                if float(max_value) <= 0.25:
                    max_value_id = speaker_total_num
                    parts = line.strip().split()
                    parts[-3] = max_value_id
                    parts = [str(part) for part in parts]
                    line = " ".join(parts) + "\n" 
                    
                # print(f'truth_speaker id: ---{max_value_id}---{max_value}')
                
                out_file.write(line.strip()+ ' ' + f'{max_value_id}' + ' ' + f'{max_value:.4f}' + '\n')
            out_file.close()
                    
    
    if args.single_read_embedding == 'True':
        input_file = []
        input_file = [f'VBx-master/rttm-txt/rttm-txt-{i:02}.txt' for i in range(1, 15+1)]
        base_path = []
        base_path = [f'VBx-master/vbx_audio/EP{i:02}/' for i in range(1, 15+1)]
        out_file = []
        out_file = [f'VBx-master/truth-rttm-txt/rttm-txt-{i:02}.txt' for i in range(1, 15+1)]
        # 寻找最大说话人标号作为其他人
        truth_path = './truth_speaker_embedding'
        truth_path = Path(truth_path)
        speaker_total_num = sum(1 for f in truth_path.iterdir() if f.is_file())
        # 演员表映射
        actor_list = './actor_list_mapping.txt'
        
        accs=[]
        for i in range(0, 15):
            logger.info(f"Extracting embedding from single file: {input_file[i]}")
            compare_embedding(input_file[i], base_path[i], './truth_speaker_embedding', out_file[i], speaker_total_num)
            acc = main(out_file[i], out_file[i], speaker_total_num)
            accs.append(acc)
            actor_list_mapping(out_file[i], out_file[i], actor_list)
        average = sum(accs) / len(accs)
        # print(f"\n-------------------平均准确率: {average:.2f}%------------------")
        logger.info("------------------Processing completed!----------------------")
        logger.info("----------- 结果保存在: VBx-master/truth-rttm-txt/ -----------")
        