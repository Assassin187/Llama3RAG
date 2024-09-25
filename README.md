<h2 id="c24y3">简介</h2>

本文使用了LangChain框架进行文本嵌入，并在Transfomers库中实现了Llama3的本地部署，可以实现本地上有文本嵌入向量数据库加强的简单问答。

<h2 id="YWwJn">软硬件依赖</h2>

本项目所使用的环境如下：

| <font style="color:rgb(31, 35, 40);">依赖项</font>           | <font style="color:rgb(31, 35, 40);">版本</font>             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <font style="color:rgb(31, 35, 40);">python</font>           | <font style="color:rgb(31, 35, 40);">3.10.12</font>          |
| <font style="color:rgb(31, 35, 40);">torch</font>            | <font style="color:rgb(31, 35, 40);">2.4.0</font>            |
| <font style="color:rgb(31, 35, 40);">transformers</font>     | <font style="color:rgb(31, 35, 40);">4.44.2</font>           |
| <font style="color:rgb(31, 35, 40);">langchain</font>        | <font style="color:rgb(31, 35, 40);">0.3</font>              |
| <font style="color:rgb(31, 35, 40);">langchain-huggingface </font> | <font style="color:rgb(31, 35, 40);">0.1.0</font>            |
| <font style="color:rgb(31, 35, 40);">faiss-gpu </font>       | <font style="color:rgb(31, 35, 40);">1.7.2</font>            |
| <font style="color:rgb(31, 35, 40);">CUDA</font>             | <font style="color:rgb(31, 35, 40);">12.2</font>             |
| <font style="color:rgb(31, 35, 40);">ubuntu</font>           | <font style="color:rgb(31, 35, 40);">22.04.3</font>          |
| <font style="color:rgb(31, 35, 40);">Linux version</font>    | <font style="color:rgb(31, 35, 40);">6.8.0-40-generic</font> |


<h2 id="jmncI">前置操作</h2>

提前下载了Llama3.1_8B模型和multilingual-e5-large嵌入模型存放于本地路径，将需要嵌入的文件放置在本地。如未下载，请在官方网站下载[Llama3.1](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)和[multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)。

<h2 id="Wrg8E">运行</h2>

```plain
python3 llama3RAG.py
```

<h2 id="taaTl">后续调整</h2>

+ 嵌入模型的选择，根据需求选择最合适的模型
+ 对于文本本身的清洗，可能能够增强效果
+ 嵌入文本切分的size和overlap
+ 检索时所使用的具体方式以及相关性阈值和返回结果数量
+ 使用RAG后生成效果的评估
+ 大模型的提示模板可以调整
+ 大模型生成时的相关系数可以影响到结果

