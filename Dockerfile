FROM gaofen/seg-farm:v0.4

# 确定容器启动时程序运行路径
WORKDIR /workspace

CMD ["/root/miniconda3/envs/gaofen/bin/python", "run.py", "/input_path", "/output_path"]