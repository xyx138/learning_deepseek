## mini_GPT

### 环境

- ubuntu22.04
- pytorch 2.5   cuda 12.4  python3.11



### 数据集

来源：[CausalLM/Refined-Anime-Text](https://huggingface.co/datasets/CausalLM/Refined-Anime-Text)

我提取了其中的中文样本，可能混有少部分的英文样本，存放在 `./dataset`，共计大约20w条数据



### 训练结果

在4090单卡花了一小时跑了一个epoch，结果如下

```bash
 Epoch: 0, Train Loss: 1.8501936064334716, Val Loss: 1.4365593758240518
```

![image-20250315115249209](https://raw.githubusercontent.com/xyx138/cloudimg/master/img/image-20250315115249209.png)



### 模型结构

![image-20250315114744816](https://raw.githubusercontent.com/xyx138/cloudimg/master/img/image-20250315114744816.png)

