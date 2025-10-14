#将rttm文件计算一列结束并输出txt文件
from pathlib import Path

def calculate_end_time(input_file, output_file):
    """
    读取输入文件，计算终止时间（起始时间 + 持续时间），并将结果写入输出文件
    
    参数:
    input_file: 输入文件名
    output_file: 输出文件名
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 跳过空行
            if not line.strip():
                outfile.write(line)
                continue
                
            # 分割行数据
            parts = line.strip().split()
            
            if len(parts) >= 5:
                try:
                    # 提取起始时间和持续时间
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    
                    # 计算终止时间
                    end_time = start_time + duration
                    
                    # 构建新的行，在持续时间后添加终止时间
                    new_line = ' '.join(parts[:5]) + f' {end_time:.6f}' + ' ' + ' '.join(parts[5:]) + '\n'
                    
                    outfile.write(new_line)
                    
                except ValueError:
                    # 如果转换失败，保持原样
                    outfile.write(line)
            else:
                # 如果格式不符合预期，保持原样
                outfile.write(line)

# 使用示例
if __name__ == "__main__":
    input_filename = []
    input_filename = [f"VBx-master/exp/EP{i:02}.rttm" for i in range(1, 15+1)]  
    output_filename = []
    output_filename = [f"VBx-master/rttm-txt/rttm-txt-{i:02}.txt" for i in range(1, 15+1)]
    
    
    for iii in range(0, 15):
        calculate_end_time(input_filename[iii], output_filename[iii])
        print(f"处理完成！结果已保存到 {output_filename[iii]}")
    print("----------- 结果保存在: VBx-master/rttm-txt/ -----------")