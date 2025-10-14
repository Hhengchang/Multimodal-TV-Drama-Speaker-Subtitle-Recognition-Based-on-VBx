import json
import re
from pathlib import Path

# 读取actor_list_mapping.txt文件并提取中文演员名
def read_actor_mapping(file_path):
    actors = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # 提取中文名（从第二个部分开始）
                chinese_name = ' '.join(parts[1:])
                actors.append(chinese_name)
    return actors

# 读取out_result01.json文件
def read_out_result(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 读取rttm-txt-01.txt文件并构建时间区间到角色ID的映射
def read_rttm_file(file_path):
    time_role_map = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 12:
                try:
                    start_time = float(parts[3])  # 转换为浮点数
                    duration = float(parts[4])    # 转换为浮点数
                    end_time = start_time + duration
                    role_id = parts[11]           # 保持为字符串
                    time_role_map.append({
                        'start': start_time,
                        'end': end_time,
                        'role_id': role_id
                    })
                except (ValueError, IndexError) as e:
                    print(f"处理rttm行时出错: {line.strip()}, 错误: {e}")
                    continue
    return time_role_map

# 将时间字符串转换为秒数
def time_to_seconds(time_str):
    # 格式: "00:02:07,383"
    time_parts = time_str.replace(',', ':').split(':')
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = int(time_parts[2])
    milliseconds = int(time_parts[3])
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    return total_seconds

# 查找时间区间对应的角色ID
def find_role_id(start_sec, end_sec, time_role_map):
    # 计算字幕段的中间时间点
    mid_point = (start_sec + end_sec) / 2.0
    
    # 首先尝试找到完全包含中间时间点的区间
    for time_span in time_role_map:
        if time_span['start'] <= mid_point <= time_span['end']:
            return time_span['role_id']
    
    # 如果没有找到完全匹配的，找最接近的
    best_match = None
    min_distance = float('inf')
    
    for time_span in time_role_map:
        # 计算时间区间中心点
        span_mid = (time_span['start'] + time_span['end']) / 2.0
        distance = abs(span_mid - mid_point)
        
        if distance < min_distance:
            min_distance = distance
            best_match = time_span['role_id']
    
    return best_match

# 主函数
def main(actor_list_mapping, out_result_json, truth_rttm_txt, result_file):
    # 读取文件
    actors = read_actor_mapping(actor_list_mapping)
    out_result_data = read_out_result(out_result_json)
    time_role_map = read_rttm_file(truth_rttm_txt)
     
    # 构建结果字典
    result_dict = {
        "roles": actors,
        "results": {}
    }
    
    # 处理每个字幕片段
    for i, item in enumerate(out_result_data):
        # 转换时间格式
        start_sec = time_to_seconds(item['start'])
        end_sec = time_to_seconds(item['end'])
        
        # 查找角色ID
        role_id = find_role_id(start_sec, end_sec, time_role_map)
        
        # 确定角色名称
        role_name = "未知角色"
        if role_id:
            # 尝试将角色ID转换为整数索引
            try:
                role_idx = int(role_id)
                if 0 <= role_idx < len(actors):
                    role_name = actors[role_idx]
                else:
                    # 如果索引超出范围，直接使用角色ID作为角色名
                    role_name = role_id
            except ValueError:
                # 如果角色ID不是数字，直接使用它
                role_name = role_id
        
        # 构建结果项
        result_item = {
            "start_time": item['start'].replace(',', '.'),
            "end_time": item['end'].replace(',', '.'),
            "text": item['text'],
            "role": role_name
        }
        
        # 添加到结果中
        result_dict["results"][str(i+1)] = result_item
    
    # 写入JSON文件
    result_file = Path(result_file)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    print(f"JSON文件已生成: {result_file}")

if __name__ == "__main__":
    actor_list_mapping = './actor_list_mapping.txt'
    out_result_json = [f'./json/out_result{i:02}.json' for i in range(1, 15+1)]
    truth_rttm_txt = [f'./VBx-master/truth-rttm-txt/rttm-txt-{i:02}.txt' for i in range(1, 15+1)]
    result_file = [f'./results/result{i:02}.json' for i in range(1, 15+1)]
    for iii in range(0, 15):
        main(actor_list_mapping, out_result_json[iii], truth_rttm_txt[iii], result_file[iii])
    print("----------- 结果保存在: ./results/ -----------")