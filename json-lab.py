from pathlib import Path
import json

output_files = "VBx-master/vad"
output_files = Path(output_files)
output_files.parent.mkdir(parents=True, exist_ok=True)


def time_to_seconds(time_str):
    # 分割小时、分钟和秒部分
    hours, minutes, seconds = time_str.split(':')
    
    # 将字符串转换为整数
    h = int(hours)
    m = int(minutes)
    
    # 将秒部分转换为秒，注意要去除毫秒部分
    s = float(seconds.replace(',', '.'))  # 将逗号替换为小数点，以便转换成浮点数

    # 计算总秒数
    total_seconds = h * 3600 + m * 60 + s
    return round(total_seconds, 3)

for iii in range(1, 15+1):
    time_split=[]
    with open(f'json/out_result{iii:02}.json', 'r') as f:
        data = json.load(f)

        for i in range(0, len(data)):
            start = time_to_seconds(data[i]['start'])
            end = time_to_seconds(data[i]['end'])
            time_split.append((start,end))

    file = output_files / f'EP{iii:02}.lab'
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, 'w') as f:
        # 初始化
        save_start, save_end = None, None
        
        for start, end in sorted(time_split, key=lambda x: x[0]):
            # 合法性检查
            if start >= end:
                continue
                
            duration = end - start
            
            # 丢弃短片段
            if duration < 0.5:
                continue
                
            # 合并邻近片段
            if save_end is not None and (start - save_end) < 0.6:
                save_end = end  # 合并
            else:
                # 写入前一段
                if save_start is not None:
                    f.write(f"{save_start:.3f} {save_end:.3f} sp\n")
                save_start, save_end = start, end
        
        # 写入最后一段
        if save_start is not None:
            f.write(f"{save_start:.3f} {save_end:.3f} sp\n")
    print(f'结果保存在 {output_files}/EP{iii:02}.lab')
    
print("----------- 结果保存在: VBx-master/vad/ -----------")
   