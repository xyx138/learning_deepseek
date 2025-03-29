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


### 加快推理

一般的推理过程，每次生成新token的时候，会重复计算之前已经计算过的权重，同时重复生成k,v矩阵。kv_cache主要做两件事：

1. 计算权重只计算当前token的q向量与之前所有token的k向量的乘积
2. 缓存下来之前算好的k向量和v向量

![kv_cache](https://raw.githubusercontent.com/xyx138/cloudimg/master/img/kv_cache.jpg)


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



## Rope

学习deepseek的**MLA**(Multi-Head Latent Attention)时候，涉及到了**rope**（Rotary Position Embedding），之间只知道sin/cos绝对位置编码，所以决定弄清楚旋转位置编码的具体过程



## 为什么需要位置编码

Transformer架构的核心就是attention机制，而attention中的重点就是计算注意力分score，即Q K 矩阵的点积过程。我们可以很容易发现，改变token的顺序，并不会影响score的值，下面的代码可以快速验证这个过程：

```python
import torch
x = torch.randn((3, 3))
x


def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()


def attention(q, k, v):
    scores = (torch.matmul(q, k.transpose(-2, -1)) / q.size(-1) ** 0.5)
    scores = [
          softmax(row)  for row in scores
        ]
    scores = torch.stack(scores, dim=0)
    return torch.matmul(scores, v)

attn_x = attention(x, x, x) 
y = x[[1, 0, 2]] # 交换第一行和第二行
attn_y = attention(y, y, y) 

attn_x, attn_y
```

结果如下：

```bash
(tensor([[-0.1147, -1.8527, -1.4377],
         [ 0.4758, -2.0564, -1.7171],
         [ 0.5365,  0.4142,  0.4669]]),
 tensor([[ 0.4758, -2.0564, -1.7171],
         [-0.1147, -1.8527, -1.4377],
         [ 0.5365,  0.4142,  0.4669]]))
```

可以发现，**score矩阵除了位置交换以外，对应的值没有改变**。但是对于一个句子来说，词的顺序改变应该会引起句意的改变，所以我们需要引入位置信息到词嵌入向量，让token顺序的改变会引起score值的改变。



## 相对位置编码

现在假设有两token：$token_m, token_n$  ，通过函数 $f(token)$将位置信息加入到向量中，简写形式$f(m)$表示将 $token_m$ 加入位置信息m

我们的目的是找到一个合适的函数 $f$ 能够在做点积运算时，能够让结果与相对位置挂钩：
$$
f(m) * f(n) = f(m - n)
$$
同时，调换token顺序结果应该不同
$$
f(m) * f(n) != f(n) * f(m)
$$
很自然想到**矩阵乘法**不可逆，似乎可以将矩阵作为映射的权重，而旋转矩阵就是我们要找的矩阵。

旋转矩阵长下面这样：
$$
R(\theta) = \left\{
 \begin{matrix}
   cos(\theta) & -sin(\theta)  \\
   sin(\theta) & sin(\theta)  \\
  \end{matrix}
  \right\}
$$
他能够将二维向量逆时针旋转 $\theta$ ，即对于 $ V' = R(\theta) V$ ，$V'$ 就是 $V$ 逆时针选择 $\theta$ 得到的向量 

容易得到：

- $R(\alpha)^T = R(-\alpha)$
- $R(\alpha) * R(\beta) = R(\alpha -\beta)$

引入旋转矩阵，两向量$q_n^{rope}, k_m^{rope}$的点积过程如下：
$$
q_n^{rope} = R(\alpha) q_n  \\

\\

k_m^{rope} = R(\beta) k_m \\

\\

\begin{aligned}({q_n^{rope}})^T * k_m^{rope} &=  (R(\alpha) q_n ) ^ T * R(\beta) k_m \\
&= q_n * R(-\alpha) * R(\beta) * k_m \\
&= q_n R(\beta - \alpha) k_m
\end{aligned}
$$
这里 $\alpha, \beta$ 与两个token的位置相对应，这样我们就能将结果与相对位置挂钩了。

扩展到多维（偶数）

![image-20250329192007988](https://raw.githubusercontent.com/xyx138/cloudimg/master/img/image-20250329192007988.png)

化简

![image-20250329192026953](https://raw.githubusercontent.com/xyx138/cloudimg/master/img/image-20250329192026953.png)



### 参考

[LLM学习记录（五）--超简单的RoPE理解方式](https://zhuanlan.zhihu.com/p/642289220)

[旋转位置编码RoPE的简单理解](https://www.bilibili.com/video/BV1CQoaY2EU2/?spm_id_from=333.337.search-card.all.click&vd_source=3a7313311adb0ce174176d9069af5bd0)

