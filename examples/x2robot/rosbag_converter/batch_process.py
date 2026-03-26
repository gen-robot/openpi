import argparse
import os
import subprocess


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="批量处理文件夹中的所有文件")
    parser.add_argument("source_dir", type=str, help="源文件夹路径，包含所有要处理的文件")
    parser.add_argument("output_dir", type=str, help="输出文件夹路径，用于存储处理后的数据")
    parser.add_argument(
        "--script_path",
        type=str,
        default="data_parse_car.py",
        help="数据解析脚本的路径，默认为当前目录下的data_parse_car.py",
    )
    parser.add_argument("--file_ext", type=str, default=".bag", help="要处理的文件扩展名，默认为.bag")
    args = parser.parse_args()

    # 获取源文件夹中所有指定扩展名的文件
    source_files = [f for f in os.listdir(args.source_dir) if f.endswith(args.file_ext)]

    if not source_files:
        print(f"在 {args.source_dir} 中未找到扩展名为 {args.file_ext} 的文件")
        return

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 遍历并处理每个文件
    for filename in source_files:
        source_file = os.path.join(args.source_dir, filename)
        print(f"正在处理文件: {filename}")
        try:
            # 执行数据解析命令
            result = subprocess.run(
                ["python3", args.script_path, source_file, args.output_dir], capture_output=True, text=True, check=True
            )
            print(f"成功处理 {filename}")
            # 打印标准输出（如果有）
            if result.stdout:
                print("脚本输出:")
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"处理文件 {filename} 时出错:")
            print(f"错误信息: {e.stderr}")
        except Exception as e:
            print(f"发生未知错误: {e!s}")


if __name__ == "__main__":
    main()
