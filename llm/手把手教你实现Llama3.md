**让我们从头开始，手把手教你，一步步实现Llama 3！**

> 参考&搬运：https://github.com/naklecha/llama3-from-scratch    
>
> 原文只给了模型代码，几乎没什么解释，并有部分错误，这里做了详细补充！
>
> 阅读本文之前，建议阅读一些Transformer的基础知识：https://jalammar.github.io/illustrated-transformer  
------

本文采用meta提供的Llama 3模型文件（非HF格式），以下是官方下载链接，选择其一即可：

（1）https://llama.meta.com/llama-downloads    

（2）https://huggingface.co/meta-llama/Meta-Llama-3-8B  

所需文件如下：

```text
Meta-Llama-3-8B
    ├── params.json
    ├── tokenizer.model
    └── consolidated.00.pth
```

------

![llama3](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_171617536.png)

# 构建Transformer的输入

## 加载分词器

> 这里不会实现BPE分词器，如果对此感兴趣，可以参考[mindbpe](https://github.com/karpathy/minbpe)，它是一个很干净的实现。    

我们直接使用OpenAI的`tiktoken`作为BPE分词器。

### 词典文件

词典文件是`tokenizer.model`，我们先打开看一下：

```text
IQ== 0
Ig== 1
Iw== 2
JA== 3
JQ== 4
...
```

😶😶😶 

可以猜测出第2列的`0, 1, 2,...`这些是词的`id`号，但是第1列是词吗，怎么几乎每个词都以`=`结尾，而且怎么全是英文？中文呢？？

我们使用`tiktoken`的`load_tiktoken_bpe`，加载这个词典：

```python
from tiktoken.load import load_tiktoken_bpe

mergeable_ranks = load_tiktoken_bpe("Meta-Llama-3-8B/tokenizer.model")
```

打印出来看一下：

```python
{
    b'!': 0,
    b'"': 1, 
    b'#': 2, 
    b'$': 3, 
    b'%': 4
    ...
}
```

再查看的源码：

```python
def load_tiktoken_bpe(
    tiktoken_bpe_file: str, expected_hash: Optional[str] = None
) -> dict[bytes, int]:
    # NB: do not add caching to this function
    contents = read_file_cached(tiktoken_bpe_file, expected_hash)
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }
```

🤣🤣🤣原来字符串"`IQ==`"是字节'`!`'的base64编码！而"`=`"是base64编码的规范要求填充的！

看来这个是Bytes-Level的BPE（Byte-Pair Encoding）分词，以1个字节为1种“字符”。

这样有个好处：

如果按照Unicode字符表进行分词，那词汇表肯定十分庞大，但是如果拆成字节，那就只有256个。

`mergeable_ranks`的前256个就是字节基础字符集，第256个之后的词都是BPE训练出来的。

### 分词器

参考meta的[llama3](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py)代码，加载分词器，代码如下：    

```python
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe

tokenizer_path = "Meta-Llama-3-8B/tokenizer.model"
special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

print(tokenizer.encode("hello world!")) # [15339, 1917, 0]
print(tokenizer.decode(tokenizer.encode("hello world!"))) # 'hello world!' 
```

😨😨😨

可以看到，里面竟然有一堆不知道干嘛的参数！

看来又得细细的拆开来看一下！！

#### special_tokens

special_tokens定义了一些特殊tokens，用以标记上下文场景。直接去看[官方文档](https://github.com/meta-llama/llama-recipes)解释：    

| Token                                          | Description                                                  |
| :--------------------------------------------- | ------------------------------------------------------------ |
| <\|begin_of_text\|>                            | Specifies the start of the prompt.                           |
| <\|end_of_text\|>                              | Specifies the end of the prompt. <br />For multiturn-conversations it's usually unused. Instead, every message is terminated with <\|eot_id\|> instead. |
| <\|eot_id\|>                                   | This token signifies the end of the message in a turn i.e. the end of a single message by a system, user or assistant role as shown below. |
| <\|start_header_id\|>{role}<\|end_header_id\|> | These tokens enclose the role for a particular message. The possible roles can be: system, user, assistant. |

Llama 3 的多轮对话遵循以下提示模板：

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ user_message_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ model_answer_1 }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ user_message_2 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

`<|eot_id|>`在开始新的标头之前，每条消息后面都会跟着一个标记，表示角色的变化。

我们举几个例子吧！

1. 系统prompt

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
2. 单个用户消息的prompt
```text
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
3. 系统prompt以及用户和助手之间的多轮对话
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Bonjour! The capital of France is Paris!<|eot_id|><|start_header_id|>user<|end_header_id|>

What can I do there?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Paris, the City of Light, offers a romantic getaway with must-see attractions like the Eiffel Tower and Louvre Museum, romantic experiences like river cruises and charming neighborhoods, and delicious food and drink options, with helpful tips for making the most of your trip.<|eot_id|><|start_header_id|>user<|end_header_id|>

Give me a detailed list of the attractions I should visit, and time it takes in each one, to plan my trip accordingly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

但是，那一堆`reserved_special_token`又是什么东西，这里没解释啊！

😨😨😨

去翻阅issue，发现有好心人去问了：[Reserved special tokens · Issue #77 · meta-llama/llama3 · GitHub](https://github.com/meta-llama/llama3/issues/77)    

总结下来就是：支持更多的`use-cases`而不需要调整词表大小。

在下游训练任务上，用户根据自己的需求，可能需要添加一些自定义的token，但是修改词表大小的成本太高，就可以改这里的`reserved_special_token`。

#### pat_str

我们先看看这段正则表达式的含义，让ChatGPT解释一下。

------
根据`|`可以分为7部分：

1. `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
   - `(?i:)` 是一个非捕获组，并且 `i` 表示忽略大小写。
   - 这部分匹配常见的英文缩略形式，如 `'s`（is 或 has），`'t`（not），`'re`（are），`'ve`（have），`'m`（am），`'ll`（will），和 `’d`（had 或 would）。

2. `[^\r\n\p{L}\p{N}]?\p{L}+`
   - `[^\r\n\p{L}\p{N}]?` 匹配一个可选的非字母、非数字、非回车、非换行符的字符。
   - `\p{L}+` 匹配一个或多个 Unicode 字符集中的字母字符。

3. `\p{N}{1,3}`
   - `\p{N}{1,3}` 匹配 1 到 3 个 Unicode 数字字符。

4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`
   - ` ?` 匹配一个可选的空格。
   - `[^\s\p{L}\p{N}]+` 匹配一个或多个非空白、非字母、非数字的字符。
   - `[\r\n]*` 匹配零个或多个回车或换行符。

5. `\s*[\r\n]+`
   - `\s*` 匹配零个或多个空白字符。
   - `[\r\n]+` 匹配一个或多个回车或换行符。

6. `\s+(?!\S)`
   - `\s+` 匹配一个或多个空白字符。
   - `(?!\S)` 是一个负向前瞻，确保后面不是一个非空白字符。

7. `\s+`
   - 匹配一个或多个空白字符。


这段正则表达式主要用于分词，它能够处理以下内容：
- 常见英文缩略形式
- 单词（由字母组成）
- 短的数字序列（1 到 3 个数字）
- 特殊符号和标点
- 空白字符和换行符

------

所以，这段正则表达式其实是一个“正则分词器”，适用于多种语言的文本处理，特别是包含多种字符集和符号的文本。

🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉


总结来看，这个tokenizer分词器包括3部分：
1. 特殊处理文本中的special_tokens
2. 用正则表达式对文本进行切分
3. 基于BPE词典对文本进行分词

## 读取模型文件

让我们先读取模型权重文件看一下：

因为模型权重文件采用的pytorch格式，直接使用`torch.load`加载即可：

```python
model = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
print(json.dumps(list(model.keys())[:20], indent=4))
```
打印出来，可以看到它是神经网络每一层的权重矩阵

```python
[
    "tok_embeddings.weight",
    "layers.0.attention.wq.weight",
    "layers.0.attention.wk.weight",
    "layers.0.attention.wv.weight",
    "layers.0.attention.wo.weight",
    "layers.0.feed_forward.w1.weight",
    "layers.0.feed_forward.w3.weight",
    "layers.0.feed_forward.w2.weight",
    "layers.0.attention_norm.weight",
    "layers.0.ffn_norm.weight",
    "layers.1.attention.wq.weight",
    "layers.1.attention.wk.weight",
    "layers.1.attention.wv.weight",
    "layers.1.attention.wo.weight",
    "layers.1.feed_forward.w1.weight",
    "layers.1.feed_forward.w3.weight",
    "layers.1.feed_forward.w2.weight",
    "layers.1.attention_norm.weight",
    "layers.1.ffn_norm.weight",
    "layers.2.attention.wq.weight"
]
```

再看一下模型配置文件：

```python
with open("Meta-Llama-3-8B/params.json", "r") as f:
    config = json.load(f)
print(config)
```

打印出来，得到模型模型的结构参数：

```python
{'dim': 4096,
 'n_layers': 32,
 'n_heads': 32,
 'n_kv_heads': 8,
 'vocab_size': 128256,
 'multiple_of': 1024,
 'ffn_dim_multiplier': 1.3,
 'norm_eps': 1e-05,
 'rope_theta': 500000.0}
```

我们使用此配置来推断模型的详细信息：

1. dim，隐藏层向量的长度为4096
2. n_layers，模型有32个Transformer层
3. n_heads，每个注意力模块有32个头
4. n_kv_heads，每个注意力模块有8个Key和Value头
5. vocab_size，词汇大小为128256
6. multiple_of，
7. ffn_dim_multiplier，
8. norm_eps，norm的防溢出eps值为1e-05
9. rope_theta，RoPE位置编码的base值

```python
dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])
```

## 编码：文本转tokens

![tokens](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_214612456.png)

我们使用上面加载的分词器`tokenizer`，将一段文本转为tokens，长度为17。

```python
prompt = "the answer to the ultimate question of life, the universe, and everything is "
# 128000对应token为<|begin_of_text|>，用来标记文本的开始
tokens = [128000] + tokenizer.encode(prompt)
print(tokens)
tokens = torch.tensor(tokens)
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens] # [17]
print(prompt_split_as_tokens)
```

```python
[128000, 1820, 4320, 311, 279, 17139, 3488, 315, 2324, 11, 279, 15861, 11, 323, 4395, 374, 220]
['<|begin_of_text|>', 'the', ' answer', ' to', ' the', ' ultimate', ' question', ' of', ' life', ',', ' the', ' universe', ',', ' and', ' everything', ' is', ' ']
```

## tokens转embedding

![embeddings](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_221650056.png)

Embedding是神经网络模块的一部分，不需要用户实现，这里直接调用`torch.nn.Embedding`即可。

```python
# 初始化Embedding模块，对应的embedding矩阵Shape为[vocab_size, dim]=[128256, 4096]
embedding_layer = torch.nn.Embedding(vocab_size, dim)
# 用model权重变量，给embedding矩阵赋值
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"]) # [128256, 4096]
# 将tokens转为embedding向量，其实就是以tokens为行索引，去抽取embedding矩阵，[17] -> [17, 4096]
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16) # [17, 4096]
```

# 构建 Transformer 的第一层

![Step of Transformer Block](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_222241997.png)

## 使用RMS对embedding归一化

![Step of rms_norm](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_222442163.png)

我们平时做归一化，有种常见的方法，就是"减去均值，除以方差"：
$$
\bar{a}_i = \frac{a_i - \mu}{\sigma}
$$

$$
\mu =\frac{1}{n}\sum_{i=1}^n{a_{i}}\quad
$$
$$
\sigma =\sqrt{\frac{1}{n}\sum_{i=1}^n{\left( a_i-\mu \right)}^2}
$$



而这里的`RMSNorm`认为，`re-centering invariance property`是不必要的，只用保留`re-scaling invariance property`，所以就把上述的均值去掉了。

最终，`RMSNorm`的计算公式如下，其中$g_{i}$是缩放因子，通过训练得到。
$$
\bar{a}_i=\frac{a_i}{\mathbf{RMS}\left( \mathbf{a} \right)}g_i,\quad \mathbf{where\,\,RMS}\left( \mathbf{a} \right) =\sqrt{\frac{1}{n}\sum_{i=1}^n{a_{i}^{2}}}
$$

为了防止分母为0，还会加一个极小值norm_eps。
```python
# def rms_norm(tensor, norm_weights):
#     rms = (tensor.pow(2).mean(-1, keepdim=True) + norm_eps)**0.5
#     return tensor * (norm_weights / rms)
def rms_norm(tensor, norm_weights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights
```
对输入embedding归一化，其中的缩放因子`norm_weights`从model权重变量中读取：

```python
token_embeddings = rms_norm(token_embeddings_unnormalized, model["layers.0.attention_norm.weight"]) # [17, 4096]
```

注意，归一化之后维度保持不变`[17, 4096]`。

## 实现Attention

![Step of attention](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_222632635.png)

我们先回忆一下，Self-Attention的计算公式：
$$
\text{Attention}\left( Q,K,V \right) \ =\ \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$
### 加载权重

让我们加载 Transformer 第一层的注意力头的权重看一下：

```python
print(
    model["layers.0.attention.wq.weight"].shape,
    model["layers.0.attention.wk.weight"].shape,
    model["layers.0.attention.wv.weight"].shape,
    model["layers.0.attention.wo.weight"].shape
)
```
`query`, `key`, `value`, `output`的权重W形状分别为：

```python
torch.Size([4096, 4096]) torch.Size([1024, 4096]) torch.Size([1024, 4096]) torch.Size([4096, 4096])
```

通过模型参数可知，这个Attention模块是一个多头注意力，应该有`n_heads=32`个头，那么就应该有32个`Wq`才对，但是这里只有1个`Wq`矩阵，完全看不到32这个值在哪，`Wk`，`Wv`,  `Wo`也同理。

这是因为，代码的作者把它们合并到1个矩阵里面了，有助于并行化注意力头的矩阵乘法运算。

### Query

![query](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_223346892.png)

#### 展开Wq

先把多头注意力的权重`Wq`展开：

```python
q_layer0 = model["layers.0.attention.wq.weight"]
head_dim = q_layer0.shape[0] // n_heads
q_layer0 = q_layer0.view(n_heads, head_dim, dim) # [32, 128, 4096]
```

得到的形状是`[32, 128, 4096]`，32是llama3中注意力头的数量，128是查询向量的大小，4096是token embedding的大小。

#### 计算Query

![Q](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_223418146.png)

1. 先拿到一个头的`Wq`，形状为`[128, 4096]`:

```python
# 以第1个头为例
q_layer0_head0 = q_layer0[0] # [128, 4096]
```

2. `Wq`与 token embedding相乘，得到token的query

```python
q_per_token = torch.matmul(token_embeddings, q_layer0_head0.T) # [17, 128]
```

可以看到，结果的形状是 `[17, 128]`，这是因为我们有 17 个token，每个token的query向量长度为128。

### 位置编码

至此，prompt对应的每个token，都有1个对应的query向量，但是如果仔细想一下就会发现，query向量在prompt的顺序位置，好像没法表达出来。

例如："the answer to the ultimate question of life, the universe, and everything is "。

在这个prompt中，“the”出现了3次，它们的位置不同，但是上述得到的query向量肯定是相同的，这导致没办法从query向量上区分它们。

所以，我们需要引入RoPE(Rotory Positional Embedding)位置编码，标记token在序列中的位置，让这3个query向量有所不同。

#### RoPE

可以看一下这段视频，学习RoPE的数学原理。https://www.youtube.com/watch?v=o29P0Kpobz0&t=530s    

![RoPE](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_223538448.png)

简单来说，就是根据token的位置，在复平面上把query向量旋转某个角度。
$$
f_{\{q,k\}}(\boldsymbol{q}_m,m)=R_{\Theta,m}^dW_{\{q,k\}}\boldsymbol{q}_m
$$

$$
\boldsymbol{R}_{\Theta,m}^d=\begin{pmatrix}\cos m\theta_1&-\sin m\theta_1&0&0&\cdots&0&0\\\sin m\theta_1&\cos m\theta_1&0&0&\cdots&0&0\\0&0&\cos m\theta_2&-\sin m\theta_2&\cdots&0&0\\0&0&\sin m\theta_2&\cos m\theta_2&\cdots&0&0\\\vdots&\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\0&0&0&0&\cdots&\cos m\theta_{d/2}&-\sin m\theta_{d/2}\\0&0&0&0&\cdots&\sin m\theta_{d/2}&\cos m\theta_{d/2}\end{pmatrix}
$$

对于向量的旋转操作，既可以使用左乘旋转矩阵R，也可以使用在复平面上乘旋转角度，为了方便理解，我们采用复数的方式表达。

😖😖😖

我们一步步来实现这个过程！

##### 计算旋转角度

为了表示复平面上的向量，我们需要把query向量两两分为一组，分别作为实部和虚部：

```python
q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2) # [17, 64, 2]
```

我们得到了一个形状为`[17, 64, 2]`的向量，它将每个token的query向量分成64个pairs！

每个pair对应复平面上的一个向量，我们需要旋转每个pairs，旋转角度为：$m\theta_{i}$，其中$m$是该token的位置，$\theta_{i}$是第$i$个pair的旋转角度。

$\theta_{i}$的计算公式如下：
$$
\theta_i=base^{-2i/d},i\in[0,1,...,d/2-1]
$$


```python
# 计算2i/d，d就是query向量的长度128，d/2就是64
zero_to_one_split_into_64_parts = torch.tensor(range(64)) / 64 # [64]
# rope_theta就是base，这个值对于模型的上下文长度外推能力有极大影响
freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts) # [64]
print(freqs)
```

```python
tensor([1.0000e+00, 8.1462e-01, 6.6360e-01, 5.4058e-01, 4.4037e-01, 3.5873e-01,
        2.9223e-01, 2.3805e-01, 1.9392e-01, 1.5797e-01, 1.2869e-01, 1.0483e-01,
        8.5397e-02, 6.9566e-02, 5.6670e-02, 4.6164e-02, 3.7606e-02, 3.0635e-02,
        2.4955e-02, 2.0329e-02, 1.6560e-02, 1.3490e-02, 1.0990e-02, 8.9523e-03,
        7.2927e-03, 5.9407e-03, 4.8394e-03, 3.9423e-03, 3.2114e-03, 2.6161e-03,
        2.1311e-03, 1.7360e-03, 1.4142e-03, 1.1520e-03, 9.3847e-04, 7.6450e-04,
        6.2277e-04, 5.0732e-04, 4.1327e-04, 3.3666e-04, 2.7425e-04, 2.2341e-04,
        1.8199e-04, 1.4825e-04, 1.2077e-04, 9.8381e-05, 8.0143e-05, 6.5286e-05,
        5.3183e-05, 4.3324e-05, 3.5292e-05, 2.8750e-05, 2.3420e-05, 1.9078e-05,
        1.5542e-05, 1.2660e-05, 1.0313e-05, 8.4015e-06, 6.8440e-06, 5.5752e-06,
        4.5417e-06, 3.6997e-06, 3.0139e-06, 2.4551e-06])
```
计算旋转角度$m\theta_{i}$：

```python
# 计算旋转角度
freqs_for_each_token = torch.outer(torch.arange(17), freqs) # [17, 64]
# 将角度转为复数
freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token) # [17, 64]

# viewing tjhe third row of freqs_cis
import matplotlib.pyplot as plt

value = freqs_cis[3]
plt.figure()
for i, element in enumerate(value[:17]):
    plt.plot([0, element.real], [0, element.imag], color='blue', linewidth=1, label=f"Index: {i}")
    plt.annotate(f"{i}", xy=(element.real, element.imag), color='red')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Plot of one row of freqs_cis')
plt.show()
```

![freqs](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_224820074.png)

##### 旋转Query

![Rotation matrix](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_223808896.png)

我们可以将query pairs转换为复数，然后使用点积来旋转：

```python
# query pairs转为复数
q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs) # [17, 64]
# 在复平面上query pairs乘以旋转角度，实现向量的旋转
q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis # [17, 64]
```

```python
# 把旋转query pairs旋转结果再转为实数
q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers_rotated) # [17, 64, 2]
# 展平query pairs，从而把query的形状恢复回去
q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape) # [17, 128]
```

至此，我们得到了带有位置编码信息的query向量

### Key

![Key](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_225028384.png)

根据Q与KV数量的对应关系，分为`1 : 1`，`n : m`, `n : 1`，分别叫做MHA（Multi-head Attention），GQA（Grouped-Query Attention），MQA（Multi-Query Attention）。

![GQA](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_225144272.png)

key的计算过程与query的计算过程几乎相同，唯一的区别是key的数量和query不同。

- 由于`n_heads=32`，`n_kv_heads=8`，query头是32个，key头是8个，那么每4个query头共享1个key头，目的是减少所需的计算次数。

- key向量也是128维
- key向量也需要添加位置信息

#### 展开Wk

```python
# 加载Wk
k_layer0 = model["layers.0.attention.wk.weight"]
# 有8个key头的Wk，每个头的Wk形状是[128, 4096]
k_layer0 = k_layer0.view(n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim) # [8, 128, 4096]
# 以第1个头为例
k_layer0_head0 = k_layer0[0] # [128, 4096]
```
#### 计算Key

![K](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_225311078.png)

```python
# 计算不带位置信息的key向量
k_per_token = torch.matmul(token_embeddings, k_layer0_head0.T) # [17, 128]
```

对key向量旋转，从而添加位置信息：


```python
# 两两一组，转为key pairs
k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2) # [17, 64, 2]
# key pairs作为复数的实部和虚部，将key pairs转为复平面上的向量
k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs) # [17, 64]
# 在复平面上key pairs乘以旋转角度，实现向量的旋转
k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis) # [17, 64, 2]
# 展平key pairs，从而把key向量的形状恢复回去
k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape) # [17, 128]
```

key的形状与query相同，都是[17, 128]，有17个token，每个token的key向量长度为128。

### Attention Map

![Attention Map](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_225426284.png)

这一步计算Attention公式中的$\text{softmax}(Q*K^T/\sqrt{d_k})$部分，可以得到一个attention map，它描述了每个token的与它之前所有token的概率依赖程度。

#### Query乘Key

先计算$Q*K^T/\sqrt{d_k}$：

```python
qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(head_dim)**0.5 # [17, 17]
```

这会得到一个score map，描述了每个token的query与每个token的key的关联程度，这就是Self-Attention。

来可视化看一下它：

```python
def display_qk_heatmap(qk_per_token):
    _, ax = plt.subplots()
    im = ax.imshow(qk_per_token.to(float).detach(), cmap='viridis')
    ax.set_xticks(range(len(prompt_split_as_tokens)))
    ax.set_yticks(range(len(prompt_split_as_tokens)))
    ax.set_xticklabels(prompt_split_as_tokens, rotation=45, rotation_mode='anchor', ha="right", va="center")
    ax.set_yticklabels(prompt_split_as_tokens)
    ax.figure.colorbar(im, ax=ax)
    
display_qk_heatmap(qk_per_token)
```

![Map](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_225549700.png)

#### Causal mask

对于模型训练，这里有个问题：

在训练期间，我们需要用过去的token去预测未来的token，但是上述score map会把未来的注意力也算进去。

例如：

对于第15个token，我们的训练目标是把第1~15个token作为上文，去预测第16个token。

然而，站在第15个token上，从因果论的角度看，第16~17个token是未来的预测目标，现在还不存在呢，所以第1~15个token与第16~17个token是没有注意力的。

于是，我们在计算attention的时候，需要屏蔽掉未来的token。

怎么屏蔽呢？可以发现：

- 对于第1个token，需要屏蔽第2~17个token
- 对于第2个token，需要屏蔽第3~17个token
- ...
- 对于第16个token，需要屏蔽第17个token
- 对于第17个token，不需要屏蔽

这样看来，对于这个17*17的score map，需要屏蔽矩阵的上三角部分。

那么我们只需要构造一个上三角mask，用它来屏蔽掉score map对应的位置，我们一般称它为causal_mask。

```python
mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
mask = torch.triu(mask, diagonal=1) # [17, 17]
print(mask)
```

```python
tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
```

```python
qk_per_token_after_masking = qk_per_token + mask  # [17, 17]
display_qk_heatmap(qk_per_token_after_masking)
```

![mask](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_225632341.png)


#### Softmax

softmax主要有2个作用：

- 归一化得到所有权重系数之和为1的概率分布
- 用softmax函数的特性突出重要元素的权重

```python
qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)  # [17, 17]
display_qk_heatmap(qk_per_token_after_masking_after_softmax)
```

![softmax](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_225723272.png)

至此，我们得到了这个attention map。

### Value

![Value](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_225834831.png)

value的计算方式和key相同，也是每4个value头之间共享权重，唯一的区别是没有位置编码。

#### 展开Wv

```python
# 加载Wv
v_layer0 = model["layers.0.attention.wv.weight"]
# 有8个value头的Wv，每个头的Wk形状是[128, 4096]
v_layer0 = v_layer0.view(n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim) # [8, 128, 4096]

# 以第1个头为例
v_layer0_head0 = k_layer0[0] # [128, 4096]
```

#### 计算Value

![V](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_225915261.png)

```python
v_per_token = torch.matmul(token_embeddings, v_layer0_head0.T) # [17, 128]
```

每个token的注意力value，形状是[17, 128]，表示有17个token，每个token的value向量长度为128。

### 计算Attention

终于可以完整的计算Attention公式了。

![Attention](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_230006680.png)

attention map中的0~1值，相当于是一个加权值，用来确定每个token需要用到多少value。

```python
qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token) # [17, 128]
```

## 多头Attention

![Step of Multi-head self attention](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_230139442.png)

多头注意力的结构如下图所示：

![Multi-head self attention](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_230552118.png)

### 计算每个头

前面提到，这个注意力模块有32个头，上面只是以第1个头为例，现在我们要写个循环，把所有头都计算出来，运算过程完全相同。

![head](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_230213701.png)

```python
qkv_attention_store = []

for head in range(n_heads):
    # 加载qkv的某个头的权重
    q_layer0_head = q_layer0[head] # [128, 4096]
    k_layer0_head = k_layer0[head//4] # [128, 4096], key weights are shared across 4 heads
    v_layer0_head = v_layer0[head//4] # [128, 4096], value weights are shared across 4 heads
    
    # 计算qkv
    q_per_token = torch.matmul(token_embeddings, q_layer0_head.T) # [17, 128]
    k_per_token = torch.matmul(token_embeddings, k_layer0_head.T) # [17, 128]
    v_per_token = torch.matmul(token_embeddings, v_layer0_head.T) # [17, 128]
	
    # 计算q的位置编码
    q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2) # [17, 64, 2]
    q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs) # [17, 64]
    q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:len(tokens)]) # [17, 64, 2]
    q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape) # [17, 128]
	
    # 计算k的位置编码
    k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2) # [17, 64, 2]
    k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs) # [17, 64]
    k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:len(tokens)]) # [17, 64, 2]
    k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape) # [17, 128]
	
    # q*kT/sqrt(d)
    qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5 # [17, 17]
    
    # mask
    mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device) # [17, 17]
    mask = torch.triu(mask, diagonal=1) # [17, 17]
    qk_per_token_after_masking = qk_per_token + mask # [17, 17]
    
    # softmax
    qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16) # [17, 17]
    
    # attention
    qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token) # [17, 128]
    
    qkv_attention_store.append(qkv_attention)

len(qkv_attention_store) # 32
```

![Heads](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_230346818.png)

我们现在有了第1层所有32个头的`qkv_attention`矩阵，接下来我要把它们合并成一个大小为 17 x 4096的大矩阵：

```python
stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1) # [17, 4096]
```

### 注意力融合

终止到了注意力模块的最后一步！

![Concatenate](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_230738771.png)

前面得到了32个头的Attention值，这里需要做一个线性变换，将他们融合起来：

```python
# 加载Wo
w_layer0 = model["layers.0.attention.wo.weight"] # [4096, 4096]
# 注意力融合
embedding_delta = torch.matmul(stacked_qkv_attention, w_layer0.T) # [17, 4096]
```

### 残差连接

我们再添加一个残差连接，即：“Attention的输入值（未norm）+ 输出值"。

![Step of skip connection](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_230816308.png)

可以使得模型更容易地学习到恒等映射，从而避免了训练深度网络时常见的梯度消失问题：

```python
embedding_after_edit = token_embeddings_unnormalized + embedding_delta # [17, 4096]
```

对结果再来一次RMS归一化：

![Step of rms_norm](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_230839159.png)

```python
embedding_after_edit_normalized = rms_norm(embedding_after_edit, model["layers.0.ffn_norm.weight"]) # [17, 4096]
```

## FFN

### SwiGLU

![Step of FFN](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_230922540.png)

在Llama3中，使用了SwiGLU激活函数的前馈网络，这种网络架构非常适合在模型需要时添加非线性。

SwiGLU到底有什么优点，这里强行解释一波：

- Swish对于负值的响应相对较小，克服了 ReLU 某些神经元上输出始终为零的缺点
- GLU 的门控特性，这意味着它可以根据输入的情况决定哪些信息应该通过、哪些信息应该被过滤。这种机制可以使网络更有效地学习到有用的表示，有助于提高模型的泛化能力。在大语言模型中，这对于处理长序列、长距离依赖的文本特别有用
- SwiGLU 中的参数可以通过训练学习，使得模型可以根据不同任务和数据集动态调整这些参数，增强了模型的灵活性和适应性
- 计算效率相比某些较复杂的激活函数（如 GELU）更高，同时仍能保持较好的性能。这对于大规模语言模型的训练和推理是很重要的考量因素

![SwiGLU](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_232619760.png)

```python
w1 = model["layers.0.feed_forward.w1.weight"] # [14336, 4096]
w2 = model["layers.0.feed_forward.w2.weight"] # [4096, 14336]
w3 = model["layers.0.feed_forward.w3.weight"] # [14336, 4096]

fc_up = torch.matmul(embedding_after_edit_normalized, w3.T) # [17, 14336]
fc_gate = torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) # [17, 14336]
output_after_feedforward = torch.matmul(fc_gate * fc_up, w2.T) # [17, 4096]
```

### 残差连接

我们再添加一个残差连接，即：“FFN的输入值（未norm）+ 输出值"。

![Step of skip connection](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_232723695.png)

可以使得模型更容易地学习到恒等映射，从而避免了训练深度网络时常见的梯度消失问题：

```python
layer_0_embedding = embedding_after_edit + output_after_feedforward  # [17, 4096]
```

# 构建所有Transformer层

Llama3一共有32层Transformer，我只需要按照同样的方式，写个for循环，实现剩下的31层就行，每一层的输入都是上一层的输出。

![All transformer layers](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_232936335.png)

```python
# embeddings input
final_embedding = token_embeddings_unnormalized

# each layer
for layer in range(n_layers):
    qkv_attention_store = []
    
    # rms_norm
    layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"]) # [17, 4096]
    
    # qkv weight for each layer
    q_layer = model[f"layers.{layer}.attention.wq.weight"] # [4096, 4096]
    q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim) # [32, 128, 4096]
    k_layer = model[f"layers.{layer}.attention.wk.weight"] # [1024, 4096]
    k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim) # [8, 128, 4096]
    v_layer = model[f"layers.{layer}.attention.wv.weight"] # [1024, 4096]
    v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim) # [8, 128, 4096]
    w_layer = model[f"layers.{layer}.attention.wo.weight"] # [4096, 4096]
    
    # each head
    for head in range(n_heads):
        # weight for head
        q_layer_head = q_layer[head] # [128, 4096]
        k_layer_head = k_layer[head//4] # [128, 4096]
        v_layer_head = v_layer[head//4] # [128, 4096]
        
        # qkv
        q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T) # [17, 128]
        k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T) # [17, 128]
        v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T) # [17, 128]
        
        # rope
        q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2) # [17, 64, 2]
        q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs) # [17, 64]
        q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis) # [17, 64, 2]
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape) # [17, 128]
        
        k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2) # [17, 64, 2]
        k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs) # [17, 64]
        k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis) # [17, 64, 2]
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape) # [17, 128]
        
         # q*kT/sqrt(d)
        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(q_per_token_rotated.shape[-1])**0.5 # [17, 17]
        
        # mask
        mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf")) # [17, 17]
        mask = torch.triu(mask, diagonal=1) # [17, 17]
        qk_per_token_after_masking = qk_per_token + mask # [17, 17]
        
        # softmax
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16) # [17, 17]
        
        # attention
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token) # [17, 128]
        qkv_attention_store.append(qkv_attention)
	
    # stack
    stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1) # [17, 4096]
    
    # attention
    w_layer = model[f"layers.{layer}.attention.wo.weight"] # [4096, 4096]
    embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T) # [17, 4096]
    
    # skip-connection
    embedding_after_edit = final_embedding + embedding_delta # [17, 4096]
    
    # rms norm
    embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"]) # [17, 4096]
    
    # ffn
    w1 = model["layers.0.feed_forward.w1.weight"] # [14336, 4096]
    w2 = model["layers.0.feed_forward.w2.weight"] # [4096, 14336]
    w3 = model["layers.0.feed_forward.w3.weight"] # [14336, 4096]
    fc_up = torch.matmul(embedding_after_edit_normalized, w3.T) # [17, 14336]
    fc_gate = torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) # [17, 14336]
    output_after_feedforward = torch.matmul(fc_gate * fc_up, w2.T) # [17, 4096]
    
    # skip-connection
    final_embedding = embedding_after_edit+output_after_feedforward
```

然后，我们对输出结果再做一次RMS归一化：

```python
final_embedding = rms_norm(final_embedding, model["norm.weight"]) # [17, 4096]
```

# 构建Transformer的输出

![Step of output](https://raw.githubusercontent.com/liangxhao/blogs/markdown/imgs/2024/06/20240622_233044019.png)

假设我们站在最后1个token，即第17个token的上，预测第18个token的概率分布，这个概率分布肯定是一个长度为vocab_size的向量，表示每个词被命中的概率。

那么，只需要做一个映射即可：

```python
logits = torch.matmul(final_embedding[-1], model["output.weight"].T) # [128256]
```

此时，`logits`是未归一化0~1之间的，如果要得到概率值，加一步`softmax`即可。

在推理时，我们只想得到概率最大值对应的token，那直接取`argmax`即可：

```python
next_token = torch.argmax(logits, dim=-1)
print(next_token)
# tensor(2983)
```

有了token之后，调用分词器解码，就可以得到文本输出：

```python
output = tokenizer.decode([next_token.item()]) 
print(output)
# '42'
```

# 结语

这只是一个十分简略的实现，按照这些步骤一步步实现，详细大家可以对Llama3以及常见的LLM的模型结构，有个直观的了解。

实际的训练和推理，肯定要经过各种性能优化，需要对代码做较大改动。