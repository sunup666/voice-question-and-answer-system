---
# 详细文档见https://modelscope.cn/docs/%E5%88%9B%E7%A9%BA%E9%97%B4%E5%8D%A1%E7%89%87
domain: #领域：cv/nlp/audio/multi-modal/AutoML
# - cv
tags: #自定义标签
-
datasets: #关联数据集
  evaluation:
  #- iic/ICDAR13_HCTR_Dataset
  test:
  #- iic/MTWI
  train:
  #- iic/SIBR
models: #关联模型
#- iic/ofa_ocr-recognition_general_base_zh

## 启动文件(若SDK为Gradio/Streamlit，默认为app.py, 若为Static HTML, 默认为index.html)
# deployspec:
#   entry_file: app.py
license: Apache License 2.0
fullWidth: true
---

# 安装依赖
确保 Asw.bat 文件中所指定的 py310 文件夹在父目录中存在（就是一个 python 3.10 的安装文件夹）
再使用 Asw.bat 用 python -m venv venv 命令创建虚拟环境
再使用 Astart.ps1 脚本激活虚拟环境，进入虚拟环境后，执行以下命令安装依赖：

```bash
 pip install -r requirements.txt
```
# 运行该 demo
在虚拟环境中使用以下命令运行该 demo：
```bash
 python app.py
```
# 模型文件夹在 models 目录下
放置好后目录结构应如下：
```bash
├── models
│   ├── Qwen
│       ├── Qwen3-0.6B/
```

具体效果图为

![检测效果图](assets/1.jpg)
