from moviepy.editor import *
import json
import numpy as np
import cv2
from rapidfuzz.distance import Levenshtein
import jieba
import os
import multiprocessing as mp
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import paddle
import threading

# 添加GPU内存监控相关的全局变量
gpu_memory_lock = threading.Lock()
gpu_memory_usage = {}

# 添加GPU优化参数
GPU_MEMORY_TARGET_PERCENT = 0.8  # 目标GPU内存使用率(80%)
GPU_MONITOR_INTERVAL = 2  # GPU监控间隔(秒)

# 视频切分处理参数
VIDEO_DIR = "./video"  # 视频路径
OUTPUT_DIR = "./json_aaaa"  # ocr预处理结果保存路径
DEBUG_DIR = "./debug"
EPOCH_COUNT = 10  # 每个视频分割的片段数
FPS = 10  # 处理帧率

# 添加线程局部存储
thread_local = threading.local()

CREDITS_KEYWORDS = ["演员表", "员表", "主演", "领衔主演", "友情出演",
                    "特别出演", "演", "饰","肖鹤云","李诗情",
                    "杜劲松","钥匙男","公交车公司办公室主任",
                    "马国强","耳机男"]  # 片尾常见关键词
MAX_CONSECUTIVE_ROLES = 4  # 连续检测到多少人名后停止处理当前视频

# 演员表提取参数
CREDITS_START_RATIO = 0.943  # 开始位置（视频总时长的百分比）
CREDITS_DURATION = 25  # 提取片尾的时长
CREDITS_FPS = 5  # 片尾提取的帧率
CREDITS_CROP = {
    "y1": 300,
    "y2": 456,
    "x1": 270,
    "x2": 440
}
CREDITS_MIN_TEXT_LENGTH = 2  # 演员表文本的最小长度
CREDITS_MIN_CONFIDENCE = 0.7  # 演员表文本的最小置信度

# 字符校正表
CHAR_CORRECTION = {
    "清": "情",
    "绣": "情",
    "统": "块",
    "作弹": "炸弹",
    "须": "报",
    "一家": "一辆",
    "都是牙": "都是汗",
    "入": "人",
    "阿娣": "阿姨",
    "洋": "样",
    "灯码": "灯吗",
    "要流纸": "耍流氓",
    "云派出所": "去派出所",
    "漫漫": "慢慢",
    "纵缠": "纠缠",
    "叔权": "叔叔",
    "社局": "杜局",
    "光梦半醒": "半梦半醒",
    "净开": "睁开",
    "办您": "办法",
    "常视": "常规",
    "源私": "循环"
}

CREDITS_CORRECTION_MAP = {
    "胡思明": "钥匙男",
    "胡匙男": "钥匙男",
    "枫倩六雷": "江枫 叶倩 小六 余雷",
    "江叶小余": "江枫 叶倩 小余",
    "翻头查看": "",
    "110控住员": "110接线员",
    "公交车公司办公": "公交车公司办公室主任",
    "公交车公司办": "公交车公司办公室主任",
    "私家车管": "私家车爸爸",
    "小余": "余雷",
    "莫医生": "莫医生",
    "刘瑶妈": "刘瑶妈妈",
    "室友": "室友",
    "杨总": "杨总",
    "枫倩": "江枫 叶倩",
    "六雷": "小六 余雷",
    "莫刘": "莫医生",
    "消防指": "消防指挥员",
    "挥员": "",
    "饰": "",
    "演": "",
    "主演": "",
    "特别出演": "",
    "友情出演": "",
    "领衔主演": "",
    "联合主演": ""
}

CREDITS_SPLIT_PATTERNS = [
    r"(.{2,3})\s*(.{2,3})\s*(.{2,3})\s*(.{2,3})",  # 匹配4个名字
    r"(.{2,3})\s*(.{2,3})\s*(.{2,3})",  # 匹配3个名字
    r"(.{2,3})\s*(.{2,3})",  # 匹配2个名字
]

# 添加角色名词典
ROLE_NAMES = [
    "肖鹤云", "李诗情", "张成", "杜劲松", "王兴德", "陶映红", "钥匙男", "马国强","药婆", "焦向荣", "耳机男",
    "江枫", "叶倩", "小六", "余雷", "莫医生","刘鹏","刘瑶", "朱师傅", "秦警官", "范永钢", "刘彩彩",
    "公交车公司办公室主任", "副市长", "私家车爸爸", "卢笛爸爸", "卢笛妈妈", "刘瑶妈妈", "于大姐", "马小龙",
    "110接线员", "吴老师", "翻斗车司机", "油罐车副驾","油罐车司机", "室友", "杨总","焦娇","男学生",
    "王兴德徒弟","司机车男孩","王萌萌","其他"
]

# 角色名合并规则
ROLE_MERGE_RULES = {
    "私家车爸爸": ["私家车", "爸爸"],
    "卢笛爸爸": ["卢笛", "爸爸"],
    "卢笛妈妈": ["卢笛", "妈妈"],
    "刘瑶妈妈": ["刘瑶", "妈妈"],
    "消防指挥员": ["消防", "指挥", "员"],
    "公交车公司办公室主任": ["公交车", "公司", "办公室", "主任"]
}

# 初始化角色名词典
for name in ROLE_NAMES:
    jieba.add_word(name)


def monitor_gpu_memory(gpu_id=0):
    """监控GPU内存使用情况"""
    try:
        # 使用paddle API获取GPU内存信息
        memory_allocated = paddle.device.cuda.memory_allocated(gpu_id) / 1024 ** 3  # 转换为GB
        memory_cached = paddle.device.cuda.memory_reserved(gpu_id) / 1024 ** 3  # 转换为GB

        with gpu_memory_lock:
            gpu_memory_usage[gpu_id] = {
                'allocated': memory_allocated,
                'cached': memory_cached,
                'timestamp': time.time()
            }

        return memory_allocated, memory_cached
    except Exception as e:
        print(f"监控GPU内存时出错: {str(e)}")
        return 0, 0


def memory_usage_monitor(interval=GPU_MONITOR_INTERVAL):
    """内存使用监控线程"""
    while True:
        try:
            with gpu_memory_lock:
                for gpu_id, usage in gpu_memory_usage.items():
                    print(f"GPU {gpu_id} 内存使用: {usage['allocated']:.2f}GB/保留: {usage['cached']:.2f}GB")

            time.sleep(interval)
        except Exception as e:
            print(f"内存监控错误: {str(e)}")
            break


def optimize_gpu_memory_usage():
    """优化GPU内存使用率"""
    print("开始优化GPU内存使用率...")

    # 启动内存监控线程
    monitor_thread = threading.Thread(target=memory_usage_monitor, daemon=True)
    monitor_thread.start()

    # 获取GPU内存信息
    total_memory = paddle.device.cuda.get_device_properties(0).total_memory / 1024 ** 3  # 转换为GB
    target_memory = total_memory * GPU_MEMORY_TARGET_PERCENT
    print(f"GPU总内存: {total_memory:.2f}GB, 目标使用: {target_memory:.2f}GB")

    # 分配内存缓冲区以提高使用率
    memory_buffers = []
    allocated_memory = 0
    chunk_size = 100 * 1024 * 1024  # 每次分配100MB

    try:
        while allocated_memory < target_memory:
            # 分配GPU内存
            buffer = paddle.rand([chunk_size // 4], dtype='float32')  # 每个float32占4字节
            memory_buffers.append(buffer)
            allocated_memory += chunk_size / 1024 ** 3  # 转换为GB

            # 监控内存使用
            current_alloc, current_cached = monitor_gpu_memory(0)
            print(f"已分配: {current_alloc:.2f}GB/目标: {target_memory:.2f}GB")

            # 短暂休眠以避免过快分配
            time.sleep(0.1)

        print(f"达到目标内存使用率: {current_alloc:.2f}GB/{target_memory:.2f}GB")

        # 保持内存占用一段时间
        time.sleep(10)

    except Exception as e:
        print(f"内存分配失败: {str(e)}")
    finally:
        # 释放内存
        print("释放GPU内存...")
        memory_buffers.clear()
        paddle.device.cuda.empty_cache()

        # 确认内存已释放
        current_alloc, current_cached = monitor_gpu_memory(0)
        print(f"释放后内存使用: {current_alloc:.2f}GB")

# # 全局OCR实例（每个进程一个）
# def init_worker(gpu_id):
#     """初始化工作进程，创建OCR实例"""
#     global ocr
#     try:
#         from paddleocr import PaddleOCR
#         # 设置当前进程使用的GPU
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#         ocr = PaddleOCR(use_textline_orientation=True, lang="ch", device='gpu')
#         print(f"进程 {os.getpid()} 初始化OCR完成，使用GPU {gpu_id}")
#     except Exception as e:
#         print(f"进程 {os.getpid()} 初始化OCR失败: {str(e)}")
#         ocr = None


def correct_text(text):
    """替换常见形近/同音字错误"""
    for wrong, right in CHAR_CORRECTION.items():
        text = text.replace(wrong, right)
    return text


def extract_white_subtitles(frame):
    """预处理字幕帧，增强白色字幕的可见性"""
    # 转换到HSV色彩空间以更好地分离颜色
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # 定义白色范围（HSV中的白色）
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # 创建白色字幕的掩码
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 应用掩码提取白色区域
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 增强对比度
    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # 转换为灰度图（OCR在灰度图上效果更好）
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

    # 应用阈值处理使字幕更清晰
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # 将单通道转回三通道（PaddleOCR需要三通道输入）
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


def text_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    distance = Levenshtein.distance(text1, text2)
    max_len = max(len(text1), len(text2))
    return 1 - (distance / max_len)  # 相似度0~1，值越高越相似


def find_overlap(text1, text2):
    """查找两个文本之间的最大重叠部分"""
    max_overlap = 0
    min_len = min(len(text1), len(text2))

    # 检查text1的结尾与text2的开头重叠
    for i in range(min_len, 0, -1):
        if text1.endswith(text2[:i]):
            return i, "suffix-prefix"

    # 检查text1的开头与text2的结尾重叠
    for i in range(min_len, 0, -1):
        if text2.endswith(text1[:i]):
            return i, "prefix-suffix"

    # 检查包含关系
    if text1 in text2:
        return len(text1), "text1-in-text2"
    if text2 in text1:
        return len(text2), "text2-in-text1"

    return 0, "no-overlap"


def merge_fragments(current, next_seg):
    """合并两个文本片段"""
    # 查找重叠部分
    overlap_len, overlap_type = find_overlap(current['text'], next_seg['text'])

    # 根据重叠类型合并文本
    if overlap_type == "suffix-prefix":
        # text1的结尾与text2的开头重叠
        merged_text = current['text'] + next_seg['text'][overlap_len:]
    elif overlap_type == "prefix-suffix":
        # text1的开头与text2的结尾重叠
        merged_text = next_seg['text'] + current['text'][overlap_len:]
    elif overlap_type == "text1-in-text2":
        # text1完全包含在text2中
        merged_text = next_seg['text']
    elif overlap_type == "text2-in-text1":
        # text2完全包含在text1中
        merged_text = current['text']
    else:
        # 没有明显重叠，选择更长的文本
        merged_text = current['text'] if len(current['text']) > len(next_seg['text']) else next_seg['text']

    return {
        'st': min(current['st'], next_seg['st']),
        'et': max(current['et'], next_seg['et']),
        'text': merged_text,
        'confidence': max(current['confidence'], next_seg['confidence'])
    }


def convert_to_json_array(file_path):
    """将每行一个JSON对象的文件转换为JSON数组格式"""
    try:
        # 读取所有行
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 如果没有数据，创建空数组
        if not lines:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('[]')
            return

        # 转换为JSON数组
        segments = [json.loads(line.strip()) for line in lines if line.strip()]

        # 按起始时间排序
        segments.sort(key=lambda x: x['st'])

        # 重新写入为JSON数组
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('[\n')
            for i, seg in enumerate(segments):
                # 最后一个元素不加逗号
                if i == len(segments) - 1:
                    f.write(f'  {json.dumps(seg, ensure_ascii=False)}\n')
                else:
                    f.write(f'  {json.dumps(seg, ensure_ascii=False)},\n')
            f.write(']')

        print(f"已转换为JSON数组格式: {file_path}")
        return len(segments)

    except Exception as e:
        print(f"转换为JSON数组时出错: {str(e)}")
        return 0


def save_segment(st, et, text, confidence, output_path):
    """保存字幕片段到文件"""
    segment = {
        'st': st,
        'et': et,
        'text': text,
        'confidence': confidence
    }
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(segment, ensure_ascii=False) + '\n')


def process_clip(st, tmp_clip, output_path, debug_dir, video_index, clip_duration, total_duration):
    """处理视频片段"""
    frame_rate = 1 / FPS
    CONFIDENCE_THRESHOLD = 0.8
    SIMILARITY_THRESHOLD = 0.4
    TIME_WINDOW = 0.7

    # 计算当前片段在整个视频中的时间范围
    clip_start_time = st
    clip_end_time = st + clip_duration

    # 计算片尾开始时间（视频总时长的90%之后）
    credits_start_time = total_duration * 0.9

    # 添加角色名检测相关变量
    consecutive_roles = 0  # 连续检测到的角色名数量
    role_detected = False  # 当前帧是否检测到角色名

    # 状态变量
    current_text = None
    start_time = None
    end_time = None
    max_confidence = 0.0

    # 确保线程局部 OCR 实例已初始化
    if not hasattr(thread_local, "ocr"):
        try:
            from paddleocr import PaddleOCR
            thread_local.ocr = PaddleOCR(use_textline_orientation=True, lang="ch", device='gpu')
            print(f"线程 {threading.get_ident()} 初始化OCR完成")
        except Exception as e:
            print(f"线程 {threading.get_ident()} 初始化OCR失败: {str(e)}")
            return False

    for cnt, cur_frame in enumerate(tmp_clip.iter_frames()):
        cur_start = frame_rate * (cnt + 1) + st
        print(f"处理第 {cnt} 帧，对应时间: {cur_start}")

        # 只在片尾部分检测角色名
        in_credits_section = cur_start >= credits_start_time

        # 检查是否已达到最大连续角色名限制（仅在片尾部分）
        if in_credits_section and consecutive_roles >= MAX_CONSECUTIVE_ROLES:
            print(f"视频 {video_index} 在片尾已连续检测到 {consecutive_roles} 个角色名，停止处理当前视频")
            return True  # 返回True表示提前终止

        try:
            if np.all(cur_frame == 0):
                continue  # 跳过全黑帧

            # 调试：保存原始帧
            if cnt % 10 == 0:
                debug_file = os.path.join(debug_dir, f"frame_{cnt}_{cur_start:.2f}.jpg")
                cv2.imwrite(debug_file, cv2.cvtColor(cur_frame, cv2.COLOR_RGB2BGR))

            # 使用线程局部的OCR实例进行识别
            result = thread_local.ocr.predict(cur_frame)
            if result and len(result[0]['rec_texts']) > 0 and len(result[0]['rec_scores']) > 0:
                text = result[0]['rec_texts'][0]
                text = correct_text(text)
                confidence = result[0]['rec_scores'][0]

                # 只在片尾部分检查角色名
                if in_credits_section:
                    role_detected = False
                    for role in ROLE_NAMES:
                        if role in text:
                            role_detected = True
                            consecutive_roles += 1
                            print(f"在片尾检测到角色名: {role}, 连续计数: {consecutive_roles}")
                            break

                    # 如果没有检测到角色名，重置计数器
                    if not role_detected and consecutive_roles > 0:
                        consecutive_roles = 0
                        print("片尾未检测到角色名，重置连续计数器")

                if confidence > CONFIDENCE_THRESHOLD:
                    if current_text is None:
                        # 初始化第一个文本
                        current_text = text
                        start_time = cur_start
                        end_time = cur_start
                        max_confidence = confidence
                    else:
                        # 计算相似度和时间差
                        sim = text_similarity(current_text, text)
                        time_diff = cur_start - start_time
                        is_continuation = (text.startswith(current_text) or current_text.startswith(text))

                        # 判断是否合并
                        if sim >= SIMILARITY_THRESHOLD or time_diff < TIME_WINDOW or is_continuation:
                            # 更新结束时间和文本
                            end_time = cur_start
                            if len(text) > len(current_text) or (
                                    len(text) == len(current_text) and confidence > max_confidence):
                                current_text = text
                                max_confidence = confidence
                        else:
                            # 保存当前文本
                            save_segment(start_time, end_time, current_text, max_confidence, output_path)
                            # 开始新文本
                            current_text = text
                            start_time = cur_start
                            end_time = cur_start
                            max_confidence = confidence

        except Exception as e:
            print(f"处理帧 {cnt} 时出错: {str(e)}")

    # 保存最后一段文本
    if current_text is not None:
        save_segment(start_time, end_time, current_text, max_confidence, output_path)

    return False  # 返回False表示正常完成

def merge_segments(output_path):
    """合并相似的字幕片段"""
    try:
        # 读取所有片段
        segments = []
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    seg = json.loads(line)
                    segments.append(seg)
                except:
                    continue

        if not segments:
            return 0

        print(f"合并前片段数: {len(segments)}")

        # 按起始时间排序
        segments.sort(key=lambda x: x['st'])

        # 合并相似片段
        merged_segments = []
        current_seg = segments[0]

        for i in range(1, len(segments)):
            next_seg = segments[i]

            # 计算时间间隔和相似度
            time_gap = next_seg['st'] - current_seg['et']
            sim = text_similarity(current_seg['text'], next_seg['text'])

            # 调试输出
            print(f"比较片段 {i - 1} 和 {i}:")
            print(f"  当前文本: '{current_seg['text']}' (st={current_seg['st']}, et={current_seg['et']})")
            print(f"  下一文本: '{next_seg['text']}' (st={next_seg['st']}, et={next_seg['et']})")
            print(f"  时间间隔: {time_gap:.2f}s, 相似度: {sim:.2f}")

            # 检查合并条件
            should_merge = False
            reason = ""

            # 条件1: 时间连续且文本相似
            if time_gap < 0.6 and sim > 0.5:
                should_merge = True
                reason = "时间连续且高度相似"

            # 条件2: 时间重叠
            elif next_seg['st'] < current_seg['et']:
                should_merge = True
                reason = "时间重叠"

            # 条件3: 文本包含关系
            elif current_seg['text'] in next_seg['text'] or next_seg['text'] in current_seg['text']:
                should_merge = True
                reason = "文本包含关系"

            # 条件4: 相同文本（即使时间间隔较大）
            elif sim > 0.95:
                should_merge = True
                reason = "文本几乎相同"

            if should_merge:
                print(f"  合并条件满足: {reason}")

                # 合并片段
                merged_seg = merge_fragments(current_seg, next_seg)

                print(f"  合并结果: st={merged_seg['st']:.2f}, et={merged_seg['et']:.2f}, text='{merged_seg['text']}'")

                # 更新当前片段
                current_seg = merged_seg
            else:
                print("  不满足合并条件")
                # 保存当前片段
                merged_segments.append(current_seg)
                current_seg = next_seg

        # 添加最后一个片段
        merged_segments.append(current_seg)

        temp_path = output_path + ".tmp"
        # 保存合并结果
        with open(output_path, 'w', encoding='utf-8') as f:
            for seg in merged_segments:
                f.write(json.dumps(seg, ensure_ascii=False) + '\n')

        os.replace(temp_path, output_path)

        print(f"合并完成: 原始片段数 {len(segments)} -> 合并后片段数 {len(merged_segments)}")
        return len(merged_segments)

    except Exception as e:
        print(f"合并片段时出错: {str(e)}")
        return 0


def extract_credits(video_path, output_path, debug_dir):
    """提取片尾演员表信息并保存为文本文件"""
    print("\n开始提取片尾演员表...")

    try:
        # 确保主线程的OCR实例已初始化
        if not hasattr(thread_local, "ocr"):
            try:
                from paddleocr import PaddleOCR
                thread_local.ocr = PaddleOCR(use_textline_orientation=True, lang="ch", device='gpu')
                print("主线程初始化OCR完成")
            except Exception as e:
                print(f"主线程初始化OCR失败: {str(e)}")
                return False

        # 加载视频
        clip = VideoFileClip(video_path)
        print(f"原始视频: FPS={clip.fps}, 时长={clip.duration:.2f}s")

        # 计算片尾开始时间
        credits_start = clip.duration * CREDITS_START_RATIO
        if credits_start < 0:
            credits_start = 0

        # 创建片尾剪辑
        credits_end = min(credits_start + CREDITS_DURATION, clip.duration)
        credits_clip = clip.subclip(credits_start, credits_end)

        # 应用特殊裁剪（覆盖全屏）
        cut_credits = credits_clip.crop(
            x1=CREDITS_CROP["x1"],
            y1=CREDITS_CROP["y1"],
            x2=CREDITS_CROP["x2"],
            y2=CREDITS_CROP["y2"]
        )
        cut_credits = cut_credits.set_fps(CREDITS_FPS)

        print(f"片尾剪辑: 时长={cut_credits.duration:.2f}s, 尺寸={cut_credits.w}x{cut_credits.h}")

        # 处理片尾帧
        frame_rate = 1 / CREDITS_FPS
        all_texts = []  # 收集所有识别到的文本

        for cnt, cur_frame in enumerate(cut_credits.iter_frames()):
            try:
                # 调试：保存原始帧
                white_frame = extract_white_subtitles(cur_frame)

                debug_file = os.path.join(debug_dir, f"credits_{cnt}.jpg")
                cv2.imwrite(debug_file, cv2.cvtColor(white_frame, cv2.COLOR_RGB2BGR))

                # 使用线程局部的OCR实例进行识别
                result = thread_local.ocr.predict(white_frame)
                if result:
                    for i, text in enumerate(result[0]['rec_texts']):
                        confidence = result[0]['rec_scores'][i]
                        if confidence > CREDITS_MIN_CONFIDENCE and len(text) >= CREDITS_MIN_TEXT_LENGTH:
                            corrected = correct_text(text)
                            all_texts.append(corrected)

            except Exception as e:
                print(f"处理片尾帧 {cnt} 时出错: {str(e)}")

        # 后处理识别结果 - 更智能的分词和合并
        processed_credits = []

        # 第一步：应用修正映射
        for text in all_texts:
            if text in CREDITS_CORRECTION_MAP:
                corrected = CREDITS_CORRECTION_MAP[text]
                if corrected:
                    # 处理需要拆分的文本
                    if " " in corrected:
                        parts = [p.strip() for p in corrected.split() if p.strip()]
                        processed_credits.extend(parts)
                    else:
                        processed_credits.append(corrected)
            else:
                processed_credits.append(text)

        # 第二步：使用jieba进行精确分词
        segmented_credits = []
        for text in processed_credits:
            # 使用自定义词典进行分词
            words = jieba.lcut(text)
            segmented_credits.extend(words)

        # 第三步：应用合并规则
        merged_credits = []
        skip_next = False

        for i in range(len(segmented_credits)):
            if skip_next:
                skip_next = False
                continue

            current = segmented_credits[i]

            # 检查是否可以与下一个词合并
            if i < len(segmented_credits) - 1:
                next_word = segmented_credits[i + 1]
                combined = current + next_word

                # 检查是否符合合并规则
                merged = False
                for full_name, parts in ROLE_MERGE_RULES.items():
                    if current in parts and next_word in parts:
                        merged_credits.append(full_name)
                        skip_next = True  # 跳过下一个词
                        merged = True
                        break

                if merged:
                    continue

            # 没有合并，单独添加当前词
            merged_credits.append(current)

        # 第四步：最终修正和过滤
        final_credits = []
        for text in merged_credits:
            # 应用最终修正映射
            if text in CREDITS_CORRECTION_MAP:
                corrected = CREDITS_CORRECTION_MAP[text]
                if corrected:
                    final_credits.append(corrected)
            else:
                # 过滤无效文本
                if len(text) >= 2 and text not in ["", "饰", "演", "饰 演"]:
                    # 检查是否可能是角色名
                    if any(name.startswith(text) or text in name for name in ROLE_NAMES):
                        final_credits.append(text)

        # 去重并保持顺序
        unique_credits = []
        for text in final_credits:
            if text not in unique_credits:
                unique_credits.append(text)

        # 保存演员表文本
        if unique_credits:
            # 写入文本文件
            with open(output_path, 'w', encoding='utf-8') as f:
                # 按顺序写入去重后的文本
                for i, text in enumerate(unique_credits):
                    f.write(f"{i} {text}\n")

            print(f"成功提取 {len(unique_credits)} 个演员表项目到: {output_path}")
        else:
            print("未识别到有效的演员表文本")

        # 关闭资源
        clip.close()
        credits_clip.close()
        cut_credits.close()

        return True

    except Exception as e:
        print(f"提取片尾演员表时出错: {str(e)}")
        return False


def process_video_multithreaded(video_info):
    """多线程处理视频的版本"""
    video_path, output_path, debug_dir, index, gpu_id = video_info

    # 设置当前线程使用的GPU
    try:
        paddle.set_device(f'gpu:{gpu_id}')
        print(f"线程 {threading.get_ident()} 使用GPU {gpu_id}")
    except Exception as e:
        print(f"无法设置GPU设备: {str(e)}，使用默认设备")

    # 原有的视频处理逻辑
    try:
        # 加载视频
        clip = VideoFileClip(video_path)
        print(f"原始视频: FPS={clip.fps}, 时长={clip.duration:.2f}s, 尺寸={clip.w}x{clip.h}")

        # 根据索引选择裁剪参数
        if index == 0:
            height = 40
            y1 = 456 - height
            y2 = 456
            x1 = 80
            x2 = 1000
        else:
            height = 35
            y1 = 440 - height
            y2 = 440
            x1 = 180
            x2 = 900

        # 应用裁剪
        cut_clip = clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)
        cut_clip = cut_clip.set_fps(FPS)
        print(f"裁剪后: FPS={cut_clip.fps}, 时长={cut_clip.duration:.2f}s, 尺寸={cut_clip.w}x{cut_clip.h}")

        # 分割视频片段
        step = cut_clip.duration / EPOCH_COUNT
        clips = []
        for i in range(EPOCH_COUNT):
            start = i * step
            end = min(start + step, clip.duration)
            if start < clip.duration:
                sub_clip = cut_clip.subclip(start, end)
                clips.append([start, sub_clip, sub_clip.duration])

        print(f"视频分割为 {len(clips)} 个片段")

        # 处理每个片段
        early_terminate = False
        for real_start, sub_clip, clip_duration in clips:
            if early_terminate:
                print("已提前终止，跳过剩余片段")
                break

            early_terminate = process_clip(
                real_start, sub_clip, output_path, debug_dir,
                index, clip_duration, clip.duration
            )

        # 合并相似片段
        merge_segments(output_path)
        convert_to_json_array(output_path)

        # 提取演员表（仅对第一个视频）
        if index == 0:
            credits_output = os.path.join(os.path.dirname(output_path), "credits.txt")
            extract_credits(video_path, credits_output, debug_dir)

        return True

    except Exception as e:
        print(f"处理视频 {video_path} 时出错: {str(e)}")
        return False
    finally:
        # 确保释放资源
        if 'clip' in locals():
            clip.close()
        if 'cut_clip' in locals():
            cut_clip.close()
        # 释放GPU内存
        try:
            paddle.device.cuda.empty_cache()
        except:
            pass

def main():
    """主函数：使用多线程处理所有视频文件"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # 优化GPU内存使用率
    optimize_gpu_memory_usage()

    # 获取所有视频文件
    video_files = sorted([f for f in os.listdir(VIDEO_DIR)
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])

    print(f"找到 {len(video_files)} 个视频文件:")
    for i, video in enumerate(video_files, 1):
        print(f"{i} {video}")

    # 准备视频处理信息
    video_infos = []
    for idx, video_file in enumerate(video_files):
        video_path = os.path.join(VIDEO_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]

        # 创建视频专属输出和调试目录
        video_output_dir = os.path.join(OUTPUT_DIR, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        output_path = os.path.join(video_output_dir, "result.json")

        video_debug_dir = os.path.join(DEBUG_DIR, video_name)
        os.makedirs(video_debug_dir, exist_ok=True)

        # 分配GPU（全部使用GPU0）
        gpu_id = 0

        video_infos.append((video_path, output_path, video_debug_dir, idx, gpu_id))

    # 使用多线程处理视频
    num_threads = 8  # 根据GPU内存调整线程数
    print(f"使用 {num_threads} 个线程并行处理视频")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_video = {
            executor.submit(process_video_multithreaded, info): info
            for info in video_infos
        }

        # 等待所有任务完成
        results = []
        for future in as_completed(future_to_video):
            video_info = future_to_video[future]
            try:
                result = future.result()
                results.append(result)
                print(f"视频处理完成: {video_info[0]}, 结果: {result}")
            except Exception as e:
                print(f"视频处理失败: {video_info[0]}, 错误: {str(e)}")
                results.append(False)

    # 输出处理结果
    success_count = sum(results)
    print(f"\n处理完成: {success_count}/{len(video_files)} 个视频成功处理")

    # 最终清理GPU内存
    try:
        paddle.device.cuda.empty_cache()
        current_alloc, current_cached = monitor_gpu_memory(0)
        print(f"最终GPU内存使用: {current_alloc:.2f}GB")
    except:
        pass

if __name__ == "__main__":
    main()