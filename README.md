## mini_GPT

### 环境

- ubuntu22.04
- pytorch 2.5   cuda 12.4  python3.11



### 数据集

来源：[CausalLM/Refined-Anime-Text](https://huggingface.co/datasets/CausalLM/Refined-Anime-Text)

我提取了其中的中文样本，可能混有少部分的英文样本, 共计大约20w条数据



### 训练结果

在4090单卡花了一小时跑了一个epoch，结果如下

```bash
 Epoch: 0, Train Loss: 1.8501936064334716, Val Loss: 1.4365593758240518
```

![image-20250315115249209](https://raw.githubusercontent.com/xyx138/cloudimg/master/img/image-20250315115249209.png)



### 模型结构

![image-20250315114744816](https://raw.githubusercontent.com/xyx138/cloudimg/master/img/image-20250315114744816.png)


## GRPO

1. 第一版

   ```
   training_args = GRPOConfig(
           output_dir='./output',
           learning_rate=5e-4,
           adam_beta1 = 0.9,
           adam_beta2 = 0.99,
           weight_decay = 0.1,
           warmup_ratio = 0.1,
           lr_scheduler_type='cosine',
           logging_steps=1,
           bf16=True,
           per_device_train_batch_size=1,
           gradient_accumulation_steps=4,
           num_generations=8,
           max_prompt_length=256,
           max_completion_length=200,
           num_train_epochs=1,
           save_steps=100,
           max_grad_norm=0.1,
           log_on_each_node=False,
           use_vllm=False,
           report_to="tensorboard"
       )
   ```

   模型最开始的表现非常棒，但是后面直接摆烂

   ![image-20250319182212302](https://raw.githubusercontent.com/xyx138/cloudimg/master/img/image-20250319182212302.png)

​	![image-20250319182340421](https://raw.githubusercontent.com/xyx138/cloudimg/master/img/image-20250319182340421.png)





2. 第二版

​	调小了初始学习率。

​	![image-20250319201457403](https://raw.githubusercontent.com/xyx138/cloudimg/master/img/image-20250319201457403.png)

但是模型似乎有些依赖digit_reward，这里比较容易获得奖励，其他方面的奖励有减少的趋势，有必有对经历函数进行优化，时模型不要过分依赖一个方面的奖励。


结果比较：

这里使用100步的检查点，比较微调前后效果。两模型（毕竟才0.5b)都不太聪明，但是微调后的模型至少可能回答正确，并且输出格式完全按照要求。

![image-20250319204555428](https://raw.githubusercontent.com/xyx138/cloudimg/master/img/image-20250319204555428.png)