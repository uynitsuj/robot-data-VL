```
git clone https://github.com/uynitsuj/robot-data-VL.git
cd robot-data-VL
git submodule update --init --recursive
conda create -n robotdatavl python==3.10
conda activate robotdatavl
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit

pip install --upgrade pip
cd dep/DeepSeek-VL2
pip install -e .
cd ../sglang
pip install -e "python[all]"
cd ../..
python scripts/example_infer_robot_data.py --model-path deepseek-ai/deepseek-vl2-tiny
```

