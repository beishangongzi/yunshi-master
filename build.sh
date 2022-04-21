# build
docker build -t gaofen/seg-farm:v0.1 .

#
docker run -itd 167747576a94
docker run  -itd -p 8022:22  --shm-size=2g --cpus=2 18d7573cc423
docker exec -it 037ad3b8b33b /bin/bash


docker run  -itd -p 2022:22  --shm-size=2g --cpus=2 77ef281caf80
docker run  -itd -p 2022:22  --shm-size=2g --cpus=2 ubuntu
docker exec -it cd96bf1c429f /bin/bash


docker tag cc12a94055ff registry.cn-beijing.aliyuncs.com/mlorry/gaofen:0.3
docker push registry.cn-beijing.aliyuncs.com/mlorry/gaofen:0.3

docker build -t hq/seg-farm:v0.2 .

# 启动项目
docker run -itd -p 8022:22 --gpus all --shm-size=2g --cpus=2 \
-v /home/huqian/data/datasets/marine-farm-seg2:/marine-farm-seg2 \
-v /home/huqian/data/datasets/marine-farm-seg:/marine-farm-seg \
-v /home/huqian/data/datasets/sample/input_path:/input_path \
-v /home/huqian/data/datasets/sample/output_path:/output_path \
gaofen/seg-farm:v0.15 /bin/bash

docker run --gpus all --shm-size=2g --cpus=2 \
-v /home/huqian/data/datasets/marine-farm-seg:/marine-farm-seg \
-v /home/huqian/data/datasets/sample/input_path:/input_path \
-v /home/huqian/data/datasets/sample/output_path:/output_path \
hq/seg-farm:v0.3




apt install libgl1-mesa-glx libglib2.0-dev
conda create -n gaofen python=3.6
conda activate gaofen
pip install pyyaml
pip install numpy pandas seaborn tqdm scikit-learn pyyaml -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install outer
pip install opencv-python -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install torch==1.2.0 torchvision==0.4.0


# 打包
docker commit 7ab3d9113096 gaofen/seg-farm:v0.18

docker build -t hq/seg-farm:v0.18 .

docker tag 29fda6aa0319 registry.cn-beijing.aliyuncs.com/mlorry/gaofen:0.18
docker push registry.cn-beijing.aliyuncs.com/mlorry/gaofen:0.18


docker run -itd -p 8022:22 --gpus all \
-v /home/huqian/data/datasets/marine-farm-seg2:/marine-farm-seg2 \
-v /home/huqian/data/datasets/marine-farm-seg:/marine-farm-seg \
-v /home/huqian/data/datasets/sample/input_path:/input_path \
-v /home/huqian/data/datasets/sample/output_path:/output_path \
gaofen/seg-farm:v0.19 /bin/bash


docker run --gpus all \
-v /home/huqian/data/datasets/marine-farm-seg2:/marine-farm-seg2 \
-v /home/huqian/data/datasets/marine-farm-seg:/marine-farm-seg \
-v /home/huqian/data/datasets/sample/input_path:/input_path \
-v /home/huqian/data/datasets/sample/output_path:/output_path \
gaofen/seg-farm:v0.18