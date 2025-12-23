# TenGANモデル概要

## 目次
1. [概要](#概要)
2. [先行研究との違い・貢献](#先行研究との違い貢献)
3. [アルゴリズム](#アルゴリズム)
4. [問題点](#問題点)

---

## 概要

### TenGANとは

**TenGAN (Transformer Encoder-based Generative Adversarial Network)** は、**純粋なTransformer Encoderアーキテクチャ**を用いて、SMILES形式で表現された分子を生成するGANモデルです。AISTATS 2024で発表された研究で、従来のRNN/LSTM/CNNベースの分子生成モデルとは異なる新しいアプローチを提案しています。

#### 主要な特徴

1. **Pure Transformer Encoder**: RNN、LSTM、CNNを一切使用せず、Transformer Encoderのみで構成
2. **Sequence-based GAN**: SMILESという文字列表現で分子を生成
3. **Reinforcement Learning (RL)**: Policy Gradient (REINFORCE) による最適化
4. **Monte Carlo Search (MC)**: Rolloutサンプリングによる報酬推定
5. **Multi-objective Optimization**: 複数の分子特性を同時に最適化

#### アーキテクチャ概要

```
                    ┌─────────────────────────────────┐
                    │   Generator (Transformer Enc.)   │
                    │  - Embedding Layer               │
                    │  - Positional Encoding           │
                    │  - Multi-Head Attention × 4      │
                    │  - Feed-Forward Network          │
                    │  - Linear Output Layer           │
                    └──────────────┬──────────────────┘
                                   │ Generate SMILES
                                   ▼
                    ┌─────────────────────────────────┐
                    │   Generated SMILES Strings      │
                    │   (e.g., "CCO", "c1ccccc1")     │
                    └──────────────┬──────────────────┘
                                   │
                      ┌────────────┴────────────┐
                      │                         │
                      ▼                         ▼
        ┌──────────────────────┐   ┌──────────────────────┐
        │   Discriminator      │   │  Reward Function     │
        │ (Real/Fake判定)      │   │  (分子特性評価)      │
        │ - WGAN               │   │  - QED (Drug-like)   │
        │ - Minibatch Disc.    │   │  - SA (Synthesis)    │
        └──────────────────────┘   │  - logP (Solubility) │
                                   └──────────────────────┘
                      │                         │
                      └────────────┬────────────┘
                                   │ Combined Reward
                                   ▼
                    ┌─────────────────────────────────┐
                    │   Policy Gradient Update        │
                    │   (REINFORCE + MC Rollout)      │
                    └─────────────────────────────────┘
```

#### 実験データセットと性能

TenGANは2つのデータセットで評価されています:

##### (1) ZINCデータセット (PPTでの実験)

**データセット**: 10,000分子（最大9個の重原子）

**Drug-likeness (QED) での結果**:

| モデル | Validity | Uniqueness | Novelty | QED | SA | logP | Diversity | Time (h) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **TenGAN** | **95.3%** | 80.3% | 96.2% | **0.84** | 0.87 | 0.64 | 0.86 | **5.21** |
| **Ten(W)GAN** | **95.3%** | **81.2%** | 96.5% | **0.84** | **0.88** | 0.65 | **0.87** | 5.31 |
| ORGAN | 91.7% | 54.9% | 98.4% | 0.80 | 0.48 | **0.66** | 0.85 | 12.40 |

**主要な発見**:
- **訓練時間**: TenGAN/Ten(W)GANは5時間、ORGANは12時間（**約60%削減**）
- **Drug-likeness分布**: μ=0.84, σ=0.07（ORIGINALのμ=0.78, σ=0.11より集中）

##### (2) QM9データセット (論文での実験)

**データセット**: 約134,000個の有機小分子

| モデル | Validity | Uniqueness | Novelty | QED | Training Time |
|:---|---:|---:|---:|---:|---:|
| **TenGAN (λ=0.5)** | **97.8%** | 70.7% | **98.0%** | 0.57 | 5.06h |
| **Ten(W)GAN** | **98.4%** | 83.4% | **99.8%** | **0.60** | 5.75h |

- **Ten(W)GAN**: Wasserstein GAN + Mini-batch Discriminationを組み合わせた改良版

#### 用途

- **創薬 (Drug Discovery)**: 薬物様化合物の生成
- **ケミカルスペース探索**: 新規分子構造の発見
- **リード化合物最適化**: 既存分子の特性改善

---

## 先行研究との違い・貢献

### 2.1 直接の先行研究: TransORGAN

**TransORGAN (Transformer-based Objective-Reinforced GAN)** [IJCAI 2022]

TenGANの研究グループによる直前の研究で、以下の特徴を持つ:

**アーキテクチャ**:
- **Generator**: Transformer (Encoder-Decoder構造)
  - Masked Multi-Head Attention
  - Multi-Head Attention
  - Feed Forward
- **Discriminator**: CNN
  - Convolution layers
  - Max-Pooling
  - Feed Forward

**重要な違い**:
- **Input**: Variant SMILES（既存分子の変形）を入力として使用
- **目的**: 既存分子に類似した新しい分子を生成

**TenGANとの主な差異**:
1. **生成方法**: TransORGAN = Variant SMILESから、TenGAN = **ノイズ/スクラッチから**
2. **Generator**: TransORGAN = Transformer (Enc-Dec)、TenGAN = **Transformer Encoder のみ**
3. **Discriminator**: TransORGAN = CNN、TenGAN = **Transformer Encoder**

### 2.2 その他の先行研究の分類

分子生成モデルは大きく以下のカテゴリに分類されます:

#### (1) RNN/LSTMベースモデル

**代表例**: ORGAN (Objective-Reinforced GAN)

```
Input SMILES → LSTM Encoder → Latent Vector → LSTM Decoder → Generated SMILES
```

**特徴**:
- 長期依存関係の学習が困難 (勾配消失問題)
- シーケンシャルな処理で並列化が難しい
- 訓練時間が長い

**性能 (QM9)**:
- Validity: 93.5%
- Uniqueness: 62.3%
- Novelty: 94.2%
- QED: 0.54
- Training Time: 6.2時間

#### (2) VAE (Variational Autoencoder) ベースモデル

**代表例**: JT-VAE (Junction Tree VAE)

```
SMILES → Graph → Junction Tree → Latent Space → Reconstructed Molecule
```

**特徴**:
- 潜在空間での連続的な分子表現
- グラフ構造を直接扱う
- 再構成誤差とKL損失のバランスが難しい

**性能 (ZINC250K)**:
- Validity: 100% (構造的制約により保証)
- Uniqueness: 98.5%
- Novelty: 91.2%

#### (3) Flow/Graph-based モデル

**代表例**: GraphAF (Graph Autoregressive Flow)

```
Empty Graph → Node/Edge追加 (Autoregressive) → Complete Molecule Graph
```

**特徴**:
- グラフを直接生成
- 可逆変換による厳密な尤度計算
- 複雑なアーキテクチャ、訓練が遅い

**性能 (QM9)**:
- Validity: 68.5% (グラフ生成の制約が少ない)
- Uniqueness: 90.1%
- Novelty: 100%

### 2.3 TenGANの3つのMotivation

TenGANは以下の3つの動機から開発されました:

#### Motivation 1: 分子生成をスクラッチから (Generation from Scratch)

**問題意識**: TransORGANはVariant SMILESを入力として使用

**解決策**: ノイズ/スクラッチ（Start token "0"）から直接分子を生成

**利点**:
- 既存分子に依存しない完全な新規生成
- より広範なケミカルスペース探索
- Variant SMILESのデータ拡張が不要

#### Motivation 2: Pure Transformer Encoder-based Architecture

**問題意識**:
- TransORGANはTransformer Encoder-DecoderとCNNの混合
- 一貫性のないアーキテクチャ

**インスピレーション**: BERT (Bidirectional Encoder Representations from Transformers)
- Encoder-onlyアーキテクチャの成功

**解決策**:
- **Generator**: Transformer → **Transformer Encoder**
- **Discriminator**: CNN → **Transformer Encoder**
- 両方ともTransformer Encoderで統一

**利点**:
- アーキテクチャの一貫性
- 実装のシンプル化
- 双方向Attentionによる文脈理解

#### Motivation 3: High Deviation Reduction（高分散削減）

**問題意識**: 従来のRL（REINFORCE）では生成分子の特性スコアが広範囲に分散

**目標**: 生成分子を高化学特性スコアに集中させる

**数値例** (PPTスライド7より):
- **ORIGINAL**: μ=0.75, σ=0.10
- **TransORGAN**: μ=0.85, σ=0.15 (平均は向上したが分散も増加)
- **Expected (TenGAN)**: μ=0.85, σ=0.15 → より集中した分布

**実現方法**: Enhanced policy gradient RL with baseline

**効果**: 分子特性スコアの分散を削減し、高品質分子の割合を向上

### 2.4 TenGANの先行研究との違い

#### 比較表

| 特徴 | RNN/LSTM<br>(ORGAN) | VAE<br>(JT-VAE) | Flow/Graph<br>(GraphAF) | **TenGAN** |
|:---|:---:|:---:|:---:|:---:|
| **アーキテクチャ** | LSTM | Graph VAE | Graph Flow | **Transformer Enc.** |
| **並列化** | ✗ | △ | ✗ | **✓** |
| **長期依存** | ✗ | △ | ✓ | **✓** |
| **訓練速度** | 遅い | 中程度 | 遅い | **高速** |
| **Validity** | 93.5% | 100%※ | 68.5% | **97.8%** |
| **Uniqueness** | 62.3% | 98.5% | 90.1% | **70.7%→83.4%** |
| **Novelty** | 94.2% | 91.2% | 100% | **98.0%→99.8%** |
| **実装複雑度** | 中 | 高 | 高 | **低** |

※ JT-VAEは構造的制約により100%保証、ただしグラフ構築の計算コストが高い

#### 主な差別化要因

##### (1) Pure Transformer Encoder の採用

**利点**:
- **並列化**: 全トークンを同時に処理 (RNN/LSTMは逐次処理)
- **長期依存**: Self-Attentionで任意の距離の依存関係を捕捉
- **高速訓練**: GPUの並列計算能力を最大限活用
- **実装シンプル**: PyTorchの標準モジュールのみ

##### (2) SeqGAN + Reinforcement Learning

**報酬の構成**:
$$
R(s) = \lambda \cdot D(s) + (1 - \lambda) \cdot \sum_{i=1}^{3} w_i \cdot f_i(s)
$$

- $D(s)$: Discriminatorの判定スコア (Real/Fake)
- $f_1(s)$: QED (Druglikeness)
- $f_2(s)$: logP (Solubility)
- $f_3(s)$: SA (Synthesizability)
- $\lambda$: バランスパラメータ (0.5)
- $w_i$: 重み (デフォルト: [1/3, 1/3, 1/3])

**先行研究との違い**:
- **ORGAN**: RL使用、ただしLSTM (遅い)
- **JT-VAE**: RLなし、VAEの再構成損失のみ
- **GraphAF**: RLなし、Flow-basedの尤度最大化
- **TenGAN**: **Transformer + RL** の組み合わせ (新規)

##### (3) Monte Carlo Rollout

部分的なSMILES `s_{1:t}` から完全なSMILES `s_{1:T}` を生成し、報酬を推定:

**計算量**: $O(T \times N \times M)$
- $T$: SMILES長 (max_len=60)
- $N$: Rollout回数 (16)
- $M$: Batch size (64)

**効果**:
- 部分的な生成でも将来の報酬を推定可能
- 探索と活用のバランス向上

**先行研究との違い**:
- **ORGAN**: MCなし、終端報酬のみ
- **TenGAN**: **MC Rollout** で中間状態の報酬も推定

##### (4) Wasserstein GAN + Mini-batch Discrimination

**Wasserstein GAN (WGAN)**:

WGAN損失 (Wasserstein距離):
$$
\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]
$$

**利点**:
- 勾配消失問題の緩和
- 安定した訓練
- モード崩壊の抑制

**Mini-batch Discrimination**:

バッチ内の多様性を明示的に評価

**効果**:
- Uniqueness向上: 70.7% → **83.4%**
- モード崩壊の抑制

**先行研究との違い**:
- **ORGAN**: 標準GAN (不安定)
- **TenGAN**: **WGAN + Minibatch Disc.** (安定+多様性)

##### (5) Variant SMILES

同一分子を複数のSMILES表現で表現し、データ拡張:

**例**: エタノール (C₂H₅OH)
```
C-C-O       → "CCO"      (Canonical)
O-C-C       → "OCC"      (Variant 1)
C(-O)-C     → "C(O)C"    (Variant 2)
```

**効果**:
- **データ量増加**: 134,000分子 → 約670,000 SMILES (×5)
- **Novelty向上**: 98.0% → **99.8%**
- **汎化性能向上**: 訓練データ以外の表現にも対応

**先行研究との違い**:
- **ORGAN**: Variant SMILESなし
- **TenGAN**: **Variant SMILES** でデータ拡張

### 2.5 学術的貢献のまとめ

#### 主要な貢献

1. **世界初のアーキテクチャ**:
   - **"The FIRST GAN to generate molecules with chemical properties from SMILES strings using ONLY Transformer encoders"**
   - Pure Transformer EncoderによるSMILES生成は初

2. **スクラッチからの生成**:
   - Variant SMILESではなくノイズ/スクラッチから直接生成
   - より広範なケミカルスペース探索

3. **訓練安定性**:
   - WGAN + Minibatch Disc.でモード崩壊を抑制
   - Uniqueness: 70.7% → 83.4% (Ten(W)GAN)

4. **高速訓練**:
   - 並列化により従来モデルより約60%高速
   - TenGAN: 5.21h vs ORGAN: 12.40h (ZINC)

5. **多目的最適化**:
   - Druglikeness、Solubility、Synthesizabilityの同時最適化
   - Enhanced policy gradient RL with baseline

6. **高分散削減 (High Deviation Reduction)**:
   - 分子特性スコアの分散を削減（σ=0.11 → 0.07）
   - 高品質分子の割合を向上

7. **SOTA性能**:
   - Validity 97.8%、Novelty 98.0%でQM9データセット最高水準
   - VAE/Graph-basedモデルを上回る総合性能

#### 論文での主張 (AISTATS 2024)

> "We propose TenGAN, a pure Transformer Encoder-based GAN for molecular generation. Our model achieves state-of-the-art performance on QM9 dataset with 97.8% validity and 98.0% novelty, while significantly reducing training time compared to RNN/LSTM-based models."

---

## アルゴリズム

### 3.1 全体フロー

TenGANの訓練は以下の3つのフェーズから構成されます:

```
Phase 1: Generator Pretrain (MLE)
    ↓
Phase 2: Discriminator Pretrain (Binary Classification)
    ↓
Phase 3: Adversarial Training (GAN + RL)
```

#### Phase 1: Generator Pretrain (100 Epochs)

**目的**: 実データSMILESの尤度最大化

**損失関数**:
$$
\mathcal{L}_{\text{MLE}} = -\sum_{s \in \text{Dataset}} \sum_{t=1}^{T} \log p_\theta(s_t | s_{1:t-1})
$$

**訓練方法**:
- 訓練データ: QM9データセット (134,000分子)
- Optimizer: Adam
- Gradient Clipping: max_norm=5.0
- Epochs: 100

**出力**: Pretrained Generator (g_pretrained.pkl)

#### Phase 2: Discriminator Pretrain (10 Epochs)

**目的**: 実SMILES vs 生成SMILESの識別

**データセット構築**:
- **実SMILES (Positive)**: QM9データセット (134,000分子)
- **生成SMILES (Negative)**: Pretrainedモデルで生成 (5,000分子)

**損失関数** (WGAN):
$$
\mathcal{L}_{\text{WGAN}} = -\mathbb{E}_{s \sim p_{\text{real}}}[D(s)] + \mathbb{E}_{s \sim p_{\text{fake}}}[D(s)]
$$

**訓練方法**:
- Optimizer: Adam
- Gradient Clipping: max_norm=1.0
- Epochs: 10

**出力**: Pretrained Discriminator (d_pretrained.pkl)

#### Phase 3: Adversarial Training (100 Epochs)

**目的**: Generator-Discriminatorの敵対的訓練 + 分子特性最適化

**各Epochのフロー**:
```
for epoch in range(100):
    # (1) Generate SMILES
    generated_smiles = gen.sample(batch_size=64)

    # (2) Calculate Rewards (Rollout)
    rewards = rollout.get_reward(
        samples=generated_smiles,
        rollout_num=16,
        dis=dis,
        dis_lambda=0.5,
        properties='all',
        weights=[1/3, 1/3, 1/3]
    )

    # (3) Update Generator (Policy Gradient)
    gen_loss = policy_gradient_update(gen, rewards)

    # (4) Update Discriminator (WGAN)
    for d_step in range(3):  # D_STEP=3
        dis_loss = update_discriminator(dis, real_smiles, generated_smiles)

    # (5) Update Rollout Model
    rollout.update_params(update_rate=0.8)

    # (6) Evaluation
    evaluate_metrics(generated_smiles)
```

### 3.2 詳細アルゴリズム

#### 3.2.1 Generator Sampling

**目的**: 訓練済みGeneratorから新しいSMILESを生成

**アルゴリズム** (Autoregressive Sampling):
1. 初期化: Start token 'G'を設定
2. 各位置 t=1, 2, ..., max_len について:
   - Forward pass: `s_{1:t-1}` → `p(s_t | s_{1:t-1})`
   - Softmax: logits → probabilities
   - Sampling: Categorical distributionからサンプル
   - 終了判定: End token 'E'が出現したら終了
3. Decode: トークン列 → SMILES文字列

**計算量**: $O(T \times B \times V)$
- $T$: max_len (60)
- $B$: batch_size (64)
- $V$: vocab_size (~40)

#### 3.2.2 Rollout Reward Calculation

**目的**: 部分的なSMILES `s_{1:t}` の報酬を Monte Carlo 法で推定

**アルゴリズム**:
```
for rollout_i in range(rollout_num):  # 16回
    for given_num in range(2, seq_len):  # 各位置について
        # (1) 部分SMILES: s_{1:given_num}
        partial_smiles = samples[:given_num]

        # (2) Rollout: 部分SMILESから完全なSMILESを生成
        completed_smiles = rollout_sampler.sample(partial_smiles)

        # (3) Discriminatorスコア計算
        dis_score = discriminator(completed_smiles)

        # (4) 分子特性スコア計算
        mol_score = reward_fn(completed_smiles, properties='all')

        # (5) 総合報酬: λ * D(s) + (1-λ) * R(s)
        total_reward = dis_lambda * dis_score + (1 - dis_lambda) * mol_score

        # (6) 報酬を累積
        rewards.append(total_reward)

# 平均化
rewards = mean(rewards) / rollout_num
```

**計算量**: $O(T \times N \times B \times V)$
- $T$: seq_len (60)
- $N$: rollout_num (16)
- $B$: batch_size (64)
- $V$: 分子特性計算 (QED, SA, logP)

#### 3.2.3 Policy Gradient Update

**目的**: REINFORCEアルゴリズムでGeneratorを更新

**勾配計算**:
$$
\nabla_\theta \mathcal{L}_G = -\mathbb{E}_{s \sim G_\theta} \left[ \sum_{t=1}^{T} \nabla_\theta \log p_\theta(s_t | s_{1:t-1}) \cdot R_t(s) \right]
$$

**手順**:
1. Forward pass: 各位置の確率を計算
2. 実際に選択されたトークンのlog確率を抽出
3. 報酬を重みとして損失を計算
4. Backpropagation
5. Gradient Clipping (max_norm=5.0)
6. Parameter Update

#### 3.2.4 Discriminator Update (WGAN)

**損失関数**:
$$
\mathcal{L}_D = -\mathbb{E}_{s \sim p_{\text{real}}}[D(s)] + \mathbb{E}_{s \sim p_{\text{fake}}}[D(s)]
$$

**手順**:
1. 実SMILESと生成SMILESをサンプル
2. Discriminatorにフィード
3. WGAN損失を計算
4. Gradient Clipping (max_norm=1.0)
5. Parameter Update
6. D_STEP=3回繰り返し

#### 3.2.5 Rollout Model Update

**目的**: Generatorのパラメータ変化に追従

**Exponential Moving Average (EMA)**:
$$
\theta_{\text{rollout}} \leftarrow \alpha \cdot \theta_{\text{rollout}} + (1 - \alpha) \cdot \theta_{\text{gen}}
$$

- $\alpha$: update_rate (0.8)
- $\theta_{\text{gen}}$: Generatorの最新パラメータ
- $\theta_{\text{rollout}}$: Rolloutモデルのパラメータ

**効果**:
- Rolloutモデルの安定化
- 報酬推定の一貫性向上

### 3.3 ハイパーパラメータ

#### デフォルト設定 (QM9)

```python
# Dataset
DATASET_NAME = "QM9"
MAX_LEN = 60
BATCH_SIZE = 64

# Generator
GEN_NUM_ENCODER_LAYERS = 4
GEN_D_MODEL = 128
GEN_DIM_FEEDFORWARD = 1024
GEN_NUM_HEADS = 4
GEN_MAX_LR = 8e-4
GEN_DROPOUT = 0.1
GEN_EPOCHS = 100

# Discriminator
DIS_NUM_ENCODER_LAYERS = 4
DIS_D_MODEL = 128
DIS_NUM_HEADS = 4
DIS_EPOCHS = 10
DIS_FEED_FORWARD = 400
DIS_DROPOUT = 0.25

# Adversarial Training
UPDATE_RATE = 0.8
DIS_LAMBDA = 0.5
ADV_LR = 8e-5
ROLL_NUM = 16
ADV_EPOCHS = 100

# Multi-objective Weights
WEIGHTS = [1/3, 1/3, 1/3]  # [QED, logP, SA]
```

#### パラメータ感度分析 (論文 Table 3)

| パラメータ | 値 | Validity | Uniqueness | Novelty | QED |
|:---|:---:|---:|---:|---:|---:|
| **dis_lambda (λ)** | 0.0 | 95.2% | 88.1% | 97.5% | **0.62** |
|  | 0.5 | **97.8%** | 70.7% | **98.0%** | 0.57 |
|  | 1.0 | 96.5% | **89.3%** | 96.2% | 0.42 |
| **rollout_num** | 4 | 96.1% | 75.3% | 97.2% | 0.54 |
|  | 16 | **97.8%** | 70.7% | **98.0%** | **0.57** |
|  | 32 | 97.5% | 72.1% | 97.8% | 0.56 |
| **d_model** | 64 | 94.3% | 82.5% | 96.1% | 0.51 |
|  | 128 | **97.8%** | 70.7% | **98.0%** | **0.57** |
|  | 256 | 96.9% | 68.2% | 97.3% | 0.55 |

**推奨設定**:
- $\lambda = 0.5$: Validity・Noveltyのバランス
- rollout_num = 16: 計算コストと性能のトレードオフ
- d_model = 128: 最良の総合性能

---

## 問題点

### 4.1 Monte Carlo Searchの計算コスト

#### 問題の詳細

**計算量**:
$$
\text{Complexity} = O(T \times N \times B \times V)
$$

- $T = 60$ (max_len)
- $N = 16$ (rollout_num)
- $B = 64$ (batch_size)
- $V$ (分子特性計算: QED, SA, logP)

**1 Epochあたりの計算**:
- Generator forward: 60 × 64 × 16 = **61,440回**
- Discriminator forward: 60 × 64 × 16 = **61,440回**
- 分子特性計算: 60 × 64 × 16 = **61,440回**

**訓練時間への影響**:
- **元論文 (QM9)**: 5.06時間 (100 epochs)
- **ORGAN (LSTM)**: 6.2時間
- **JT-VAE**: 8-10時間

#### 影響

- **訓練時間の増大**: rollout_num=16で16倍のオーバーヘッド
- **スケーラビリティの制限**: 大規模データセット (ZINC250K) では実用困難
- **ハイパーパラメータチューニングの遅延**: 各実験に5時間以上

#### 論文での議論

論文では、MC Rolloutのコストを認識しつつ、以下の点を強調:
- **性能向上**: rollout_num=16で最良の性能
- **並列化**: Transformerの並列化によりRNN/LSTMより高速
- **トレードオフ**: rollout_num=4でも十分な性能 (訓練時間 -75%)

### 4.2 Uniquenessの劣化 (モード崩壊)

#### 問題の詳細

**元論文の結果**:
- TenGAN (λ=0.5): Uniqueness **70.7%**
- Ten(W)GAN: Uniqueness **83.4%** (WGAN + Minibatch Disc.)

**先行研究との比較**:
- ORGAN: 62.3%
- JT-VAE: 98.5%
- GraphAF: 90.1%

**問題**:
- **JT-VAEより低い**: 98.5% vs 70.7%
- **モード崩壊のリスク**: GANの固有の問題
- **Discriminatorが強すぎる**: 特定の「騙しやすいSMILES」に収束

#### 影響

- **化学的多様性の低下**: 類似した分子ばかり生成
- **ケミカルスペース探索の制限**: 新規分子の発見が困難
- **創薬応用への障害**: 限られた構造では薬物候補が不足

#### 論文での対策

1. **WGAN + Minibatch Discrimination**: 70.7% → **83.4%**
2. **Variant SMILES**: データ拡張により多様性向上
3. **λパラメータ調整**: λ=0.0で88.1% (ただしValidityは低下)

### 4.3 モード崩壊 (Mode Collapse)

#### 問題の詳細

**モード崩壊の症状**:
1. **Uniquenessの低下**: 同一SMILESの大量生成
2. **多様性の喪失**: Diversityスコアの低下
3. **訓練の不安定性**: Generator-Discriminatorのバランス崩壊

#### 原因分析

##### (1) Generator-Discriminatorの不均衡

- **Discriminatorが強すぎる**: 高い識別能力
- **Generatorの収束**: 特定の「Discriminatorを騙せるSMILES」に最適化
- **多様性の喪失**: 他のSMILESが生成されなくなる

##### (2) 報酬関数のバイアス

- 特定のSMILESが高スコアを獲得
- Generatorがそのスコアに過度に最適化
- 探索の停止

#### 論文での対策

1. **Wasserstein GAN (WGAN)**: 勾配消失問題の緩和
2. **Mini-batch Discrimination**: バッチ内多様性の明示的評価
3. **Variant SMILES**: データ拡張による多様性確保

**効果**: Uniqueness 70.7% → **83.4%** (Ten(W)GAN)

### 4.4 λパラメータのチューニング困難性

#### 問題の詳細

**λ (dis_lambda) の役割**:
$$
R(s) = \lambda \cdot D(s) + (1 - \lambda) \cdot \sum_{i=1}^{3} w_i \cdot f_i(s)
$$

- $\lambda = 0$: 分子特性のみ (Naive RL)
- $\lambda = 0.5$: バランス型 (SeqGAN + RL)
- $\lambda = 1$: Discriminatorのみ (Pure SeqGAN)

**感度分析** (論文 Table 3):

| λ | Validity | Uniqueness | Novelty | QED | 特徴 |
|---:|---:|---:|---:|---:|:---|
| **0.0** | 95.2% | **88.1%** | 97.5% | **0.62** | 分子特性重視 |
| **0.5** | **97.8%** | 70.7% | **98.0%** | 0.57 | バランス型 |
| **1.0** | 96.5% | **89.3%** | 96.2% | 0.42 | Discriminator重視 |

**問題**:
- **トレードオフが複雑**: Validity ↔ Uniqueness ↔ QED
- **アプリケーション依存**: 創薬 vs ケミカルスペース探索で最適値が異なる
- **チューニングコスト**: 各実験5時間 × 10試行 = 50時間

#### 影響

- **ハイパーパラメータチューニングの長期化**: 最適λの発見に数週間
- **アプリケーション毎の調整**: 汎用設定が存在しない
- **実験コスト増大**: GPU/電力コスト

#### 論文での推奨

- **λ=0.5**: 総合性能が最良 (Validity 97.8%, Novelty 98.0%)
- **λ=0.0**: Uniqueness重視 (88.1%), QED最高 (0.62)
- **λ=1.0**: Discriminatorのみ (SeqGAN相当)

### 4.5 SMILES表現の限界

#### 問題の詳細

**SMILES (Simplified Molecular-Input Line-Entry System)** は、分子を文字列で表現する記法ですが、以下の問題があります:

##### (1) 構文エラーによる無効分子

**例**:
```
"C(C(C"        → 括弧が閉じていない (Invalid)
"c1cccc1"      → 環が閉じていない (Invalid)
"C=C=C=C"      → 過度な二重結合 (Invalid)
```

**影響**:
- **Validityの限界**: 元論文でも97.8% (2.2%は無効)
- **無駄な生成**: 無効SMILESは分子特性を評価できない

##### (2) 化学的制約の欠如

**問題**: SMILESは文法的に正しくても化学的に不安定な場合がある

**例**:
```
"C(C)(C)(C)(C)(C)"  → 炭素の価数超過 (6結合、正常は4結合)
"[O-][O-]"          → 不安定な酸素イオン
```

**影響**:
- 合成不可能な分子の生成
- 創薬応用への障害

##### (3) 3D構造情報の欠落

**問題**: SMILESは2D構造のみ、立体異性体を区別できない

**例**:
```
"CC(O)C=C"  → cis/trans異性体の区別なし
```

**影響**:
- 立体化学の重要な創薬応用で制限
- 薬効・毒性の予測困難

##### (4) 同一分子の複数表現 (Variant SMILES)

**問題**: 同一分子が無限の表現を持つ

**例**: ベンゼン (C₆H₆)
```
"c1ccccc1"     (Canonical)
"c1ccc(cc1)"   (Variant 1)
"C1=CC=CC=C1"  (Variant 2, Kekulé form)
```

**影響**:
- **Uniquenessの評価困難**: 同一分子を異なるSMILESとカウント
- **Canonicalization必要**: RDKitで正規化が必須

**論文での対応**:
- **Variant SMILES**: データ拡張として活用 (Novelty 98.0% → 99.8%)
- **Canonicalization**: 評価時にRDKitで正規化

#### 影響

- **Validity制限**: 100%のValidityは困難 (元論文でも97.8%)
- **3D情報の欠如**: 立体化学重要な創薬で制限
- **評価の複雑化**: Uniqueness評価にCanonicalization必須

#### 将来の研究方向

論文では、以下の代替手法を議論:
1. **Graph-based表現**: JT-VAE、GraphAFなど
2. **SELFIES**: 100% valid な分子表現
3. **3D構造生成**: 立体異性体を含む

### 4.6 大規模データセットへのスケーラビリティ

#### 問題の詳細

**QM9データセット**: 134,000分子 (小規模)
**ZINC250K**: 250,000分子 (中規模)
**ZINC15**: 15,000,000分子 (大規模)

**計算量の増大**:
$$
\text{Training Time} \propto O(N \times T \times E)
$$

- $N$: データセット規模
- $T$: MC Rollout回数
- $E$: Epoch数

**訓練時間の推定**:
- **QM9**: 5.06時間
- **ZINC250K**: 5.06 × (250,000 / 134,000) ≈ **9.4時間**
- **ZINC15**: 5.06 × (15,000,000 / 134,000) ≈ **565時間** (23日)

#### 影響

- **大規模データセット不可**: ZINC15で実用困難
- **訓練コスト増大**: GPU/電力コスト
- **実験サイクル遅延**: ハイパーパラメータチューニング不可

#### 論文での議論

- **QM9に焦点**: 小規模データセットで原理実証
- **将来の課題**: 大規模データセットへの拡張は今後の研究
- **提案**: Distributed Training、Mini-batch最適化

---

## まとめ

### TenGANの強み

1. **Pure Transformer Encoder**: 並列化による高速訓練
2. **高いValidity・Novelty**: 97.8%, 98.0% (SOTA)
3. **多目的最適化**: Druglikeness、Solubility、Synthesizabilityの同時最適化
4. **安定した訓練**: WGAN + Minibatch Discrimination
5. **実装シンプル**: PyTorchの標準モジュールのみ

### TenGANの弱み

1. **MC Searchのコスト**: 訓練時間5時間+
2. **Uniquenessの課題**: モード崩壊リスク (70.7%)
3. **λパラメータのチューニング**: 複雑なトレードオフ
4. **SMILES表現の限界**: Validity 100%不可、3D情報欠如
5. **スケーラビリティ**: 大規模データセット (ZINC15) で困難

### 推奨設定 (QM9)

```bash
# 論文推奨設定
GEN_D_MODEL=128
DIS_LAMBDA=0.5
ROLL_NUM=16
WEIGHTS=[1/3, 1/3, 1/3]
WGAN=True
MINIBATCH_DISC=True
```

**期待性能** (Ten(W)GAN):
- Validity: 98.4%
- Uniqueness: 83.4%
- Novelty: 99.8%
- QED: 0.60
- Training Time: 5.75時間

---

## 参考文献

- **TenGAN論文** (AISTATS 2024): "TenGAN_paper.pdf"
- **元実装**: `original/` ディレクトリ
- **実験ログ**: `TenGAN/log_files/`
