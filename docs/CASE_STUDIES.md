# 案例叙事：API 决策 Token 的逐层涌现机制

> 基于 `case_study_candidates.json`（logit lens，sharpest rank-jump per bucket）
> 模型：StarCoder2-7B base；共 32 个 decoder 层（层 0 = 第一层）
> 所有案例均来自 `full_span_correct` 子集：模型在无提示情况下完整预测出正确 API span

---

## 背景：涌现层的分布特征

在 376 个正确样本中，`emergence_layer_top10`（logit lens）的分布高度集中：

- 中位数：**第 18 层**（共 32 层，约 56% 深度处）
- P25–P75：第 18–20 层
- 分桶阈值（< 18 / 18–20 / > 20）对应"早于典型 / 典型 / 晚于典型"三组

三个案例分别代表三个分桶，展示不同的涌现路径。

---

## 案例一：库生态系统辨别（典型涌现，mid bucket）

**样本**：`qwencoder-3b-1389`  
**任务**：预测 `tf.random.categorical`（`tensorflow.random.categorical` 的别名形式）  
**决策 token**：`'tf'`（API span 的首 token）  
**emergence_layer_top10**：18 &nbsp;|&nbsp; **saturation_layer**：25 &nbsp;|&nbsp; rank jump at emergence：**1509**

### 上下文（决策点前的关键代码）

```python
obs_ph = tf.compat.v1.placeholder(shape=(None, obs_dim), dtype=tf.float32)
logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])

# make action selection op (outputs int actions, sampled from policy)
actions = tf.squeeze(  # <-- 决策点
```

模型需要在 `tf.squeeze(` 之后的 `\n    actions = ` 位置，预测下一个 API 调用的命名空间前缀。

### 逐层轨迹（logit lens）

| 层 | rank | prob | JSD-to-final | top-3 候选 |
|---|---|---|---|---|
| 16 | 1744 | ≈0 | 0.674 | `'random'`, `' random'`, `'Random'` |
| 17 | 1516 | ≈0 | 0.675 | `' arg'`, `'argmax'`, `'ables'` |
| **18** | **7** | 0.005 | 0.661 | `'slim'`, `'jax'`, `'oso'` |
| 19 | 3 | 0.013 | 0.639 | `' tf'`, `'slim'`, `'tf'` |
| 20 | 2 | 0.153 | 0.457 | `' tf'`, `'tf'`, `'slim'` |
| 25 | 1 | — | — | (saturation) |

### 叙事

**层 16–17（"知道该做什么，不知道用谁"）**：早期层输出 `'random'`、`' arg'`、`'argmax'` 等"函数动作"类 token——模型已经感知到"这是个采样/随机操作"，但还没锁定是哪个库的 API。

**层 18（"框架定位"）**：rank 从 1516 跌至 7，单步跌幅 **1509**——这是全数据集中最剧烈的一次涌现跳变。此时 top-3 出现 `'slim'`（TF-Slim 子模块）、`'jax'`（Google JAX）——两者都是 TF 生态系统的成员！正确 token `'tf'` 虽然还未进 top-3，但模型已经缩小范围到"Google ML 生态"。

**层 19–20（"命名空间收敛"）**：`' tf'` 跃居 top-1，`'slim'`（另一个 TF 子命名空间）仍作为竞争候选出现，说明模型在这两层完成了从"TF 生态" → "标准 `tf.` 命名空间"的最终辨别。

**层 25（saturation）**：rank 稳定为 1，预测锁定。

**关键机制**：中间层先识别"生态系统归属"（slim/jax 等协同框架 token 涌现），再在更深层收敛到具体命名空间缩写——这是一种分层语义缩减过程，不是单步记忆提取。

---

## 案例二：框架竞争解析（早期涌现，shallow bucket）

**样本**：`codegemma-2b-357`  
**任务**：预测 `torch.linalg.lstsq`  
**决策 token**：`'torch'`  
**emergence_layer_top10**：17 &nbsp;|&nbsp; **saturation_layer**：21 &nbsp;|&nbsp; rank jump at emergence：**502**

### 上下文（决策点前的关键代码）

```python
# 注意：prompt 里同时出现了 TensorFlow 版本！
model2 = tf.linalg.lstsq(partial_X_train, partial_y_train[...,np.newaxis], fast=False).numpy()

case "pytorch-qrcp":
    start_lstsq = timer()
    model = np.array(  # <-- 决策点（np.array 是 completion_prefix，真正预测点是此之后）
```

Prompt 里明确出现了 `tf.linalg.lstsq`——一个 TensorFlow 版本的同名 API。模型必须在"case 'pytorch-qrcp'"语境下，从 `tf.linalg.lstsq` 的干扰中识别出正确的 `torch.linalg.lstsq`。

### 逐层轨迹（logit lens）

| 层 | rank | prob | JSD-to-final | top-3 候选 |
|---|---|---|---|---|
| 15 | 1306 | ≈0 | 0.645 | `'ody'`, `'Exact'`, `'Handlers'` |
| 16 | 505 | ≈0 | 0.641 | `'?'`（garbled）, `'orb'`, `'ody'` |
| **17** | **3** | 0.011 | 0.615 | `' torch'`, `'?'`, `'torch'` |
| 18 | 2 | 0.044 | 0.564 | `' torch'`, `'torch'`, `'ody'` |
| 21 | 1 | — | — | (saturation) |

### 叙事

**层 15–16（"杂讯主导"）**：早期层充斥着 BPE 杂讯 token（`'?'`、`'ody'`、`'orb'`），完全没有语义相关内容。这是 logit lens 视角下早期层的常见现象——残差流尚未建立起语义表征。

**层 17（"框架竞争解决"）**：rank 从 505 跌至 3，top-3 中出现 `' torch'` 和 `'torch'`——两种 tokenization 形式的 torch 已经同时跃升！注意：prompt 里明明有 `tf.linalg.lstsq`，但模型在层 17 就已经锁定 `torch`，说明 `case "pytorch-qrcp"` 的上下文标签足以在这一层覆盖 TF 版本的干扰。

**层 18–20（小幅震荡）**：rank 在 1–2 之间震荡，saturation 最终在层 21 实现。这种"已接近但需要几层微调"的模式在 shallow bucket 很典型。

**关键机制**：单一的大语义跳变（层 17，rank 降幅 502），且已出现正确 token 而非 TF 竞争对手。这表明模型的框架辨别（从"有 tf 版本"到"选 torch 版本"）是一个在极少层内完成的"开关"，可能与 `case "pytorch-qrcp"` 语境标签的 direct attention 传播有关。

---

## 案例三：语义层次组装（晚期涌现，deep bucket）

**样本**：`codegemma-2b-764`  
**任务**：预测 `GaussianMixture`（`sklearn.mixture.GaussianMixture` 的直接类名形式）  
**决策 token**：`'GaussianMixture'`  
**emergence_layer_top10**：21 &nbsp;|&nbsp; **saturation_layer**：31 &nbsp;|&nbsp; rank jump at emergence：**125**

### 上下文（决策点前的关键代码）

```python
# GMM hyper-parameter: (1) Proportional to #regions (2) clipped to 1~10
n_components = np.clip(total_num // 100, a_min=1, a_max=10)
# Construct Gaussian Mixture Model for source feature and target feature
print(total_num, flush=True)
if total_num > 1:
    src_gm =   # <-- 决策点（赋值右侧第一个 token）
```

模型需要在赋值 `src_gm = ` 之后预测 `GaussianMixture(n_components=...)`——一个仅出现一次的 sklearn 类名，无法通过简单的近邻 copy 预测（`'GaussianMixture'` 未在 prompt 的可见上文中出现）。

### 逐层轨迹（logit lens）

| 层 | rank | prob | JSD-to-final | top-3 候选 |
|---|---|---|---|---|
| 19 | 761 | ≈0 | 0.625 | `' make'`, `' mixture'`, `'sk'` |
| 20 | 131 | 0.001 | 0.610 | `' np'`, `'sk'`, `' mixture'` |
| **21** | **6** | 0.009 | 0.556 | `' mixture'`, `'sk'`, `' Mix'` |
| 22 | 6 | 0.019 | 0.509 | `' mixture'`, `'mix'`, `'gm'` |
| 23 | 3 | 0.029 | 0.463 | `' mixture'`, `'mix'`, `' Gaussian'` |
| 31 | 1 | — | — | (saturation) |

### 叙事

**层 19（"领域感知"）**：top-3 出现 `' mixture'`（混合模型操作）和 `'sk'`（sklearn 缩写）——模型在第 19 层就已经识别到"这是 sklearn 混合模型领域的调用"，只是尚未锁定具体类名。`' make'` 的出现暗示模型还在考虑 factory-style 构造（如 `make_pipeline`）。

**层 20（"库归属确认"）**：`'sk'` 和 `' np'` 竞争，`' mixture'` 维持——模型在 sklearn 和 numpy 之间摇摆（高斯混合模型在两个库中均有实现：`sklearn.mixture.GaussianMixture` vs 手写 numpy 版）。

**层 21（"类型分辨"）**：rank 从 131 跌至 6（涌现点）。top-3 出现 `' Mix'`——这是 `GaussianMixture` 类名的**词根形式**，提示模型正在组装大写首字母的类名而非小写函数名。`'gm'` 出现于层 22（`GaussianMixture` 的常见缩写），进一步确认。

**层 23（"'Gaussian'出现"）**：`' Gaussian'` 首次出现在 top-3——模型开始明确地指向"高斯"分布的具体实现，而非泛化的"混合"概念。

**层 31（saturation，极晚）**：整个过程跨越 10 层以上，是典型的"深层涌现 + 极晚 saturation"模式。JSD 在层 21 时仍高达 0.556，说明预测分布与最终层分布差异极大——模型在这 10 层里持续精炼。

**关键机制**：这是一个"语义层次组装"的典型案例。模型先感知领域（混合模型），再锁定库（sklearn），再锁定类型（类名 vs 函数名），再锁定具体词（Gaussian），最终在最后一层 saturation。每一步均可从 top-3 候选的变化中观察到。这与 Geva et al. (2022) 所描述的 FFN"逐层构建词汇预测"机制高度吻合。

---

## 全候选汇总表

| 分桶 | row_id（简） | library | ground_truth_api | em_top10 | sat | jump |
|------|------------|---------|-----------------|---------|-----|------|
| **shallow** | codegemma-2b-898 | tensorflow | global_variables_initializer | 17 | 18 | 504 |
| shallow | codegemma-2b-357 | pytorch | torch.linalg.lstsq | 17 | 21 | 502 |
| shallow | codegemma-2b-625 | pytorch | torch.linalg.lstsq | 17 | 21 | 502 |
| shallow | codegemma-2b-859 | tensorflow | global_variables_initializer | 17 | 18 | 309 |
| **mid** | qwencoder-3b-1389 | tensorflow | random.categorical | 18 | 25 | 1509 |
| mid | starcoder2-7b::tf::… | tensorflow | tensorflow.print | 20 | 31 | 1446 |
| mid | codegemma-2b-80 | pytorch | torch.linalg.cholesky | 18 | 20 | 593 |
| mid | codegemma-2b-84 | pytorch | torch.linalg.qr | 19 | 19 | 536 |
| mid | deepseek-1.3b-904 | sklearn | GaussianMixture | 19 | 19 | 406 |
| **deep** | qwencoder-3b-417 | pytorch | nn.functional.interpolate | 21 | 28 | 146 |
| deep | codegemma-2b-764 | sklearn | GaussianMixture | 21 | 31 | 125 |
| deep | deepseek-1.3b-903 | sklearn | GaussianMixture | 21 | None | 103 |
| deep | qwencoder-3b-514 | pytorch | nn.functional.interpolate | 21 | 29 | 70 |
| deep | codegemma-2b-821 | tensorflow | random.categorical | 22 | 29 | 26 |

---

## 跨案例归纳

三个案例揭示出三种不同的涌现路径，但有共同结构特征：

1. **早期层输出与语义无关**：所有案例的早期层（< 15 层）均充斥杂讯 token，从 logit lens 视角几乎无法解读；tuned lens 能在同层读出更多语义信号（参见 `emergence_summary.json`）。

2. **涌现总是"跨越式"的**：rank 不是线性递减，而是先缓慢降（数百名开外），某层突然跳变入 top-10（rank-jump 中位数约 400+），此后再经若干层微调收敛至 rank-1。这与 nostalgebraist (2020) 的"迭代精炼（iterative refinement）"假说一致。

3. **pre-emergence 竞争画像有层次**：
   - "生态系统层"：与正确库同属一个技术生态的 token（`'slim'`/`'jax'` 竞争 `'tf'`）
   - "语义函数层"：语义接近但非具体 API 的 token（`' mixture'`/`'gm'` 竞争 `'GaussianMixture'`）
   - "框架竞争层"：prompt 中明确出现的另一框架版本（`'tf.linalg.lstsq'` 竞争 `'torch.linalg.lstsq'`）
   
   三类竞争分别对应"库归属辨别"、"语义组装"、"上下文指代消歧"三种不同的计算需求。

4. **saturation 普遍滞后于 emergence**：涌现（进 top-10）比饱和（锁定 rank-1）通常早 5–10 层，说明模型在"知道答案"之后还需要数层来"确认答案"——这一"确认窗口"对应 JSD 从高位缓慢降低至接近 0 的过程。

---

*数据来源：`output/starcoder2_7b/topk_trajectory.jsonl`（logit lens，376 个正确样本，97.9s on GPU 3）*
