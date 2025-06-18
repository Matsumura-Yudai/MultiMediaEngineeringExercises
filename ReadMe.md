---
title: "マルチメディア工学演習"
numbering: true
output:
  html_document:
    toc: true
    number_sections: true
---

[TenGAN: Pure Transformer Encoders Make an Efficient Discrete GAN for De Novo Molecular Generation | Chen Li, Yoshihiro Yamanishi Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, PMLR 238:361-369, 2024.]:https://proceedings.mlr.press/v238/li24d.html
[TenGAN ソースコード]:https://github.com/naruto7283/TenGAN
[WSL Anaconda導入]:https://www.salesanalytics.co.jp/datascience/datascience141/

# マルチメディア工学演習について
生成AIを活用したゼロからの創薬に取り組む予定です。
具体的には、以下の論文で提案されている手法（Transformer Encoder＋GAN＋強化学習を用いたSMILESの単一属性最適化）を基に、複数属性の最適化を目指します。

## 目次
- [参考文献](#参考文献)
- [TenGANについて](#tenganについて)
    - [背景と課題](#背景と課題)
    - [提案手法（TenGAN / Ten(W)GAN）](#提案手法tengan--tenwgan)
    - [実験と結果](#実験と結果)
    - [Contributeとまとめ](#contributeとまとめ)
    - [今後の課題](#今後の課題)
- [ローカル](#ローカル)
    - [環境構築](#環境構築)
        - [Anaconda](#Anaconda)
        - [仮想環境構築](#仮想環境構築)
    - [既存のプログラムの実行](#既存のプログラムの実行)
        - [ZINC](#zinc)
        - [QM9](#qm9)
            - [事前学習](#事前学習)
            - [敵対的学習](#敵対的学習)
- [演習内容](#演習内容)
    - [単純和](#単純和)
- [20250517](#20250517)

# 参考文献
1. [TenGAN: Pure Transformer Encoders Make an Efficient Discrete GAN for De Novo Molecular Generation | Chen Li, Yoshihiro Yamanishi Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, PMLR 238:361-369, 2024.]
2. [TenGAN ソースコード]
3. [WSL Anaconda導入]

# TenGANについて
## 背景と課題
- 新しい分子（化合物）をゼロから自動生成（de novo molecular generation）する研究は、創薬などで非常に重要です。

- 特に分子を文字列で表現するSMILESフォーマットを使った生成には、GAN（敵対的生成ネットワーク）が使われますが、
離散データを扱うため、学習が不安定になりやすく、モード崩壊や多様性の低下が問題になっていました。

## 提案手法（TenGAN / Ten(W)GAN）
- TenGANは、純粋にTransformer Encoderのみで構成された新しいGANモデルです。

- ジェネレータもディスクリミネータも、従来のRNNやCNNではなく、マスク付きTransformer Encoderを使います。

- 強化学習（Reinforcement Learning） を組み合わせることで、離散データに対しても学習可能にしています。

- バリアントSMILES（同じ分子を違う順番で表すSMILES） を使ったデータ拡張で、SMILESの意味・構文をより深く学習。

- 改良版のTen(W)GANではさらに、

    - ミニバッチディスクリミネーション（多様性向上）

    - Wasserstein GAN（学習安定化） を導入しています。

## 実験と結果
- 小規模な分子データセット（QM9, ZINC）で実験。

- 有効な分子の生成率（Validity）、新規性（Novelty）、多様性（Diversity）など、ほぼ全ての指標で既存手法（ORGANなど）を上回る成果。

- 生成分子の薬物らしさ（QEDスコア）、合成容易性（SAスコア）、溶解性（logP）の最適化にも成功。

- 従来のRNNベースGANより、学習速度も高速（GPUでの実行時間が短縮）。

## Contributeとまとめ
- 完全にTransformer EncoderだけでSMILES文字列を生成できる新しいGANを開発。

- データ拡張・ミニバッチディスクリミネーション・WGANを組み合わせ、学習安定性と多様性を両立。

- 生成された分子が、化学的にも意味があり、より高品質であることを実証。

## 今後の課題
モンテカルロサーチ（強化学習中に使う）による計算コストと性能安定性のトレードオフが残っており、
将来的にはSoft Actor-Critic法などを使った改善を検討している。

# ローカル
[参考文献[2]](https://github.com/naruto7283/TenGAN)を参考にして、TenGANの実行環境をローカルに構築し、実験等を行う。

## 環境構築
[参考文献[2]](https://github.com/naruto7283/TenGAN)のREADMEを参考にして、環境構築を行う。
なお、著者の開発環境は以下のとおりである。
- Windows11 HOME
- wsl2(Ubuntu)

### Anaconda
[参考文献[3]](https://www.salesanalytics.co.jp/datascience/datascience141/)を参考にして、以下の手順でWSL(Ubuntu)にAnacondaをインストールした。

1. 最新版のAnacondaをダウンロード
[Anaconda公式ダウンロードサイト](https://www.anaconda.com/products/distribution#Downloads)にアクセスし、最新のLinux版のAnacondaのリンクをコピーして、
Ubuntu上で以下のコマンドを実行する。
ただし、以下のURLは2025年4月現在、最新版のものである。
```
$ wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
```

2. Anacondaをインストール
Ubuntu上で、以下のコマンドを記述し実行すると、ダウンロードしたAnacondaのインストールが開始される。
ただし、シェルスクリプトはURLの末尾に示されたものである。
```
$ bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

3. condaコマンドのパスを通す
ホームディレクトリの下にある「.bashrc」ファイルを開き、末尾に
```bash
export PATH="/home/[User Name]/anaconda3/bin:$PATH"
```
※[User Name]はUbuntuのユーザー名
を追加し、一度、ターミナルを再起動するか、
```
$ source .bashrc
```
というコマンドを実行する。
その後、
```
$ conda --version
```
というコマンドを実行し、インストールしたcondaのバージョンが表示されれば、Anaconda環境の構築は完了である。

### 仮想環境構築
[TenGAN ソースコード]にはTenGAN実行のための仮想環境が付属している。
これをアクティベートするには、以下の手順でコマンドを実行する。

1. ソースコードの一部を書き換え
付属しているソースコードは一部、書き換えが必要な箇所が存在する。
具体的には、TenGAN/env/env.ymlの117行目
```yml
- install==1.3.5
```
である。これをコメントアウトするか、削除する必要がある。

また、TenGAN/main.pyの36行目の
```python
parser.add_argument('--gen_pretrain:', action='store_true', help='whether pretrain the dataset')
```
のうち```'--gen_pretrain:'```の末尾の:(コロン)が不要なため、元の行を削除するかコメントアウトして、以下のように書き換える。
```python
parser.add_argument('--gen_pretrain', action='store_true', help='whether pretrain the dataset')
```

2. 環境の構築と起動
TenGANディレクトリに移動し、以下のコマンドを順に実行すれば、仮想環境の構築は完了である。
```
$ conda env create -n tengan_env -f env/env.yml
$ source ~/.bashrc
$ conda activate tengan_env
```
ターミナルの画面において以下のように表示されていれば成功である。
```
(tengan_env) UserName:~/ ... /TenGAN$
```

## 既存のプログラムの実行
[TenGAN ソースコード]に付属のmain.pyを実行してみた。

### ZINC
実行方法は、TenGANディレクトリに移動し、仮想環境が起動した状態で以下のコマンドを実行するだけである。
```
$ python main.py
```
実行結果は以下の通りである。
```
(tengan_env) you2002724@UD0724:~/workspace/MultiMediaEngineeringExercises/TenGAN$ python main.py



Vocabulary Information:
==================================================================
{' ': 0, '^': 1, '$': 2, 'H': 3, 'B': 4, 'c': 5, 'C': 6, 'n': 7, 'N': 8, 'o': 9, 'O': 10, 'p': 11, 'P': 12, 's': 13, 'S': 14, 'F': 15, 'Q': 16, 'W': 17, 'I': 18, '[': 19, ']': 20, '+': 21, 'Z': 22, 'X': 23, '-': 24, '=': 25, '#': 26, '.': 27, '(': 28, ')': 29, '1': 30, '2': 31, '3': 32, '4': 33, '5': 34, '6': 35, '7': 36, '@': 37, '/': 38, '\\': 39}


Parameter Information:
==================================================================
POSITIVE_FILE            :   dataset/ZINC.csv
NEGATIVE_FILE            :   res/generated_smiles_ZINC.csv
G_PRETRAINED_MODEL       :   res/save_models/ZINC/TenGAN_0.5/rollout_8//batch_64/druglikeness/g_pretrained.pkl
D_PRETRAINED_MODEL       :   res/save_models/ZINC/TenGAN_0.5/rollout_8//batch_64/druglikeness/d_pretrained.pkl
PROPERTY_FILE            :   res/save_models/ZINC/TenGAN_0.5/rollout_8//batch_64/druglikeness/trained_results.csv
BATCH_SIZE               :   64
MAX_LEN                  :   70
VOCAB_SIZE               :   40
DEVICE                   :   cuda
GPUS                     :   1


GEN_PRETRAIN             :   False
GENERATED_NUM            :   10000
GEN_TRAIN_SIZE           :   9600
GEN_NUM_ENCODER_LAYERS   :   4
GEN_DIM_FEEDFORWARD      :   1024
GEN_D_MODEL              :   128
GEN_NUM_HEADS            :   4
GEN_MAX_LR               :   0.0008
GEN_DROPOUT              :   0.1
GEN_EPOCHS               :   150


DIS_PRETRAIN             :   False
DIS_WGAN                 :   False
DIS_MINIBATCH            :   False
DIS_NUM_ENCODER_LAYERS   :   4
DIS_D_MODEL              :   100
DIS_NUM_HEADS            :   5
DIS_MAX_LR               :   8e-07
DIS_EPOCHS               :   10
DIS_FEED_FORWARD         :   200
DIS_DROPOUT              :   0.25


ADVERSARIAL_TRAIN        :   False
PROPERTIES               :   druglikeness
DIS_LAMBDA               :   0.5
MODEL_NAME               :   TenGAN_0.5
UPDATE_RATE              :   0.8
ADV_LR                   :   8e-05
G_STEP                   :   1
D_STEP                   :   1
ADV_EPOCHS               :   100
ROLL_NUM                 :   8
==================================================================



Start time is 2025-04-27 14:35:25



GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs


Load Pre-trained Generator.
Generating 10000 samples...
100%|█████████████████████████████████████████████████████████████████████| 156/156 [01:58<00:00,  1.32it/s]

Results Report:
********************************************************************************
Total Mols:   9984
Validity:     8496    (85.10%)
Uniqueness:   8310    (97.81%)
Novelty:      8108    (97.57%)
Diversity:    0.89


Samples of Novel SMILES:
CC(=O)c1ccc(CNC(=O)[C@@H]2C=CC[C@@H]2C(=O)[O-])cc1
O=C(COC(=O)c1cnc(Cl)c(Cl)c1)Nc1ccc(Cl)cc1Cl
O=C(Nc1ccc(Cl)c(Cl)c1)[C@@H]1[C@@H]2C=C[C@@H](O2)[C@@H]1C(=O)[O-]
C[C@@H](Oc1ccccc1)C(=O)Nc1cccc(S(N)(=O)=O)c1
O=C(CSc1nnc2ccccn12)NCCCc1ccc(Cl)cc1


[druglikeness]: [Mean: 0.788   STD: 0.104   MIN: 0.271   MAX: 0.946]
********************************************************************************


GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs


Load Pre-trained Discriminator.
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs


Load TenGAN Generator: res/save_models/ZINC/TenGAN_0.5/rollout_8//batch_64/druglikeness/Epoch_66_gen.pkl


Generating 10000 samples...
100%|█████████████████████████████████████████████████████████████████████| 156/156 [00:58<00:00,  2.67it/s]

Results Report:
********************************************************************************
Total Mols:   9983
Validity:     9514    (95.30%)
Uniqueness:   7943    (83.49%)
Novelty:      7647    (96.27%)
Diversity:    0.87


Samples of Novel SMILES:
CCC(=O)Oc1ccccc1C[NH+]1CCOCC1
O=C(Cc1ccccc1)Nc1ccccc1Br
CCOc1ccccc1C[NH+]1[C@@H](C)CCC[C@H]1C
COc1ccc(C#N)cc1OCC(=O)Nc1ccccc1
COC(=O)c1ccccc1C(=O)Nc1ccccc1


[druglikeness]: [Mean: 0.830   STD: 0.081   MIN: 0.246   MAX: 0.946]
********************************************************************************


Load TenGAN Discriminator: res/save_models/ZINC/TenGAN_0.5/rollout_8//batch_64/druglikeness/Epoch_66_dis.pkl


Top-12 Molecules of [druglikeness]:
c1cc(NC(=O)c2ccccc2Br)c(C(=O)[O-])cc1    0.946
c1cc(S(=O)(Nc2ccc3c(c2)OCO3)=O)ccc1F     0.945
N(c1ccc2c(c1)OCO2)S(=O)(c1ccc(F)cc1)=O   0.945
c1(NC(c2ccc(Cl)c(Cl)c2)=O)ccccc1C(=O)[O-]        0.945
c1(C(Nc2ccc(C(=O)[O-])cc2)=O)c(Cl)cccc1Cl        0.945
C(=O)([O-])c1ccccc1NC(c1ccc(Cl)cc1Cl)=O          0.945
c1(NC(c2ccc(Cl)cc2Cl)=O)ccc(C([O-])=O)cc1        0.945
N(S(=O)(c1ccc2c(c1)OCO2)=O)c1ccc(F)cc1F          0.944
O1c2ccccc2OC[C@@H]1C(Nc1ccc(N(C)C)cc1)=O         0.944
c1ccc2c(c1)O[C@@H](C(Nc1ccc(N(C)C)cc1)=O)CO2     0.944
C1CCCc2c1nn(CC(=O)Nc1ncc(Cl)cc1)c2       0.944
O=C(Cn1c(=O)c2c(cccc2)nc1CC)NC1CCCC1     0.944
********************************************************************************


File names for drawing distributions: ['res/generated_smiles_ZINC.csv']
Distributions are not generated.
********************************************************************************
```

### QM9
データセットQM9.csvを用いて実行した。

#### 事前学習
まずは、生成器と識別機の事前学習を行った。
```
(tengan_env) you2002724@UD0724:~/workspace/MultiMediaEngineeringExercises/TenGAN$ python main.py --gen_pretrain --dis_pretrain --dataset_name QM9 --max_len 60 --generated_num 5000 --gen_train_size 4800 --roll_num 16



Vocabulary Information:
==================================================================
{' ': 0, '^': 1, '$': 2, 'H': 3, 'B': 4, 'c': 5, 'C': 6, 'n': 7, 'N': 8, 'o': 9, 'O': 10, 'p': 11, 'P': 12, 's': 13, 'S': 14, 'F': 15, 'Q': 16, 'W': 17, 'I': 18, '[': 19, ']': 20, '+': 21, 'Z': 22, 'X': 23, '-': 24, '=': 25, '#': 26, '.': 27, '(': 28, ')': 29, '1': 30, '2': 31, '3': 32, '4': 33, '5': 34, '6': 35, '7': 36, '@': 37, '/': 38, '\\': 39}


Parameter Information:
==================================================================
POSITIVE_FILE            :   dataset/QM9.csv
NEGATIVE_FILE            :   res/generated_smiles_QM9.csv
G_PRETRAINED_MODEL       :   res/save_models/QM9/TenGAN_0.5/rollout_16//batch_64/druglikeness/g_pretrained.pkl
D_PRETRAINED_MODEL       :   res/save_models/QM9/TenGAN_0.5/rollout_16//batch_64/druglikeness/d_pretrained.pkl
PROPERTY_FILE            :   res/save_models/QM9/TenGAN_0.5/rollout_16//batch_64/druglikeness/trained_results.csv
BATCH_SIZE               :   64
MAX_LEN                  :   60
VOCAB_SIZE               :   40
DEVICE                   :   cuda
GPUS                     :   1


GEN_PRETRAIN             :   True
GENERATED_NUM            :   5000
GEN_TRAIN_SIZE           :   4800
GEN_NUM_ENCODER_LAYERS   :   4
GEN_DIM_FEEDFORWARD      :   1024
GEN_D_MODEL              :   128
GEN_NUM_HEADS            :   4
GEN_MAX_LR               :   0.0008
GEN_DROPOUT              :   0.1
GEN_EPOCHS               :   150


DIS_PRETRAIN             :   True
DIS_WGAN                 :   False
DIS_MINIBATCH            :   False
DIS_NUM_ENCODER_LAYERS   :   4
DIS_D_MODEL              :   100
DIS_NUM_HEADS            :   5
DIS_MAX_LR               :   8e-07
DIS_EPOCHS               :   10
DIS_FEED_FORWARD         :   200
DIS_DROPOUT              :   0.25


ADVERSARIAL_TRAIN        :   False
PROPERTIES               :   druglikeness
DIS_LAMBDA               :   0.5
MODEL_NAME               :   TenGAN_0.5
UPDATE_RATE              :   0.8
ADV_LR                   :   8e-05
G_STEP                   :   1
D_STEP                   :   1
ADV_EPOCHS               :   100
ROLL_NUM                 :   16
==================================================================



Start time is 2025-04-27 16:43:06



GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs


Pre-train Generator...
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Epoch 149: 100%|███████████████████| 79/79 [00:02<00:00, 37.42it/s, loss=0.534, v_num=2, val_loss=0.891]
Generator Pre-train Time: 0.25 hours
Generating 5000 samples...
100%|███████████████████████████████████████████████████████████████████| 78/78 [00:14<00:00,  5.26it/s]

Results Report:
********************************************************************************
Total Mols:   4992
Validity:     4471    (89.56%)
Uniqueness:   4183    (93.56%)
Novelty:      3493    (83.50%)
Diversity:    0.92


Samples of Novel SMILES:
NC1C2NC1C(O)C2O
NC(=O)CC1(O)COC1
CCCC12NC3C1C32O
CC(CO)C(=O)C(=N)N
O=Cn1nnc(CO)n1


[druglikeness]: [Mean: 0.477   STD: 0.069   MIN: 0.175   MAX: 0.665]
********************************************************************************


GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs


Pre-train Discriminator...
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Epoch 9: 100%|█| 157/157 [00:02<00:00, 77.58it/s, loss=0.699, v_num=3, val_loss=0.693, val_acc=0.519, tr
Discriminator Pre-train Time: 0.27 hours
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs


TenGAN Generator path does NOT exist: res/save_models/QM9/TenGAN_0.5/rollout_16//batch_64/druglikeness/Epoch_66_gen.pkl
```

#### 敵対的学習
事前学習した生成器と識別機を用いて敵対的学習および生成を行った。
```
(tengan_env) you2002724@UD0724:~/workspace/MultiMediaEngineeringExercises/TenGAN$ python main.py --adversarial_train --dataset_name QM9 --max_len 60 --generated_num 5000 --gen_train_size 4800 --roll_num 16



Vocabulary Information:
==================================================================
{' ': 0, '^': 1, '$': 2, 'H': 3, 'B': 4, 'c': 5, 'C': 6, 'n': 7, 'N': 8, 'o': 9, 'O': 10, 'p': 11, 'P': 12, 's': 13, 'S': 14, 'F': 15, 'Q': 16, 'W': 17, 'I': 18, '[': 19, ']': 20, '+': 21, 'Z': 22, 'X': 23, '-': 24, '=': 25, '#': 26, '.': 27, '(': 28, ')': 29, '1': 30, '2': 31, '3': 32, '4': 33, '5': 34, '6': 35, '7': 36, '@': 37, '/': 38, '\\': 39}


Parameter Information:
==================================================================
POSITIVE_FILE            :   dataset/QM9.csv
NEGATIVE_FILE            :   res/generated_smiles_QM9.csv
G_PRETRAINED_MODEL       :   res/save_models/QM9/TenGAN_0.5/rollout_16//batch_64/druglikeness/g_pretrained.pkl
D_PRETRAINED_MODEL       :   res/save_models/QM9/TenGAN_0.5/rollout_16//batch_64/druglikeness/d_pretrained.pkl
PROPERTY_FILE            :   res/save_models/QM9/TenGAN_0.5/rollout_16//batch_64/druglikeness/trained_results.csv
BATCH_SIZE               :   64
MAX_LEN                  :   60
VOCAB_SIZE               :   40
DEVICE                   :   cuda
GPUS                     :   1


GEN_PRETRAIN             :   False
GENERATED_NUM            :   5000
GEN_TRAIN_SIZE           :   4800
GEN_NUM_ENCODER_LAYERS   :   4
GEN_DIM_FEEDFORWARD      :   1024
GEN_D_MODEL              :   128
GEN_NUM_HEADS            :   4
GEN_MAX_LR               :   0.0008
GEN_DROPOUT              :   0.1
GEN_EPOCHS               :   150


DIS_PRETRAIN             :   False
DIS_WGAN                 :   False
DIS_MINIBATCH            :   False
DIS_NUM_ENCODER_LAYERS   :   4
DIS_D_MODEL              :   100
DIS_NUM_HEADS            :   5
DIS_MAX_LR               :   8e-07
DIS_EPOCHS               :   10
DIS_FEED_FORWARD         :   200
DIS_DROPOUT              :   0.25


ADVERSARIAL_TRAIN        :   True
PROPERTIES               :   druglikeness
DIS_LAMBDA               :   0.5
MODEL_NAME               :   TenGAN_0.5
UPDATE_RATE              :   0.8
ADV_LR                   :   8e-05
G_STEP                   :   1
D_STEP                   :   1
ADV_EPOCHS               :   100
ROLL_NUM                 :   16
==================================================================



Start time is 2025-04-27 17:03:37



GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs


Load Pre-trained Generator.
Generating 5000 samples...
100%|███████████████████████████████████████████████████████████████████| 78/78 [01:07<00:00,  1.16it/s]

Results Report:
********************************************************************************
Total Mols:   4992
Validity:     4459    (89.32%)
Uniqueness:   4149    (93.05%)
Novelty:      3456    (83.30%)
Diversity:    0.92


Samples of Novel SMILES:
COc1cc(CO)no1
CC1C2C[N][C]3OC1N32
N#CC12CC1C1CC12
O=C1CC12CCOC2=O
CCOc1oncc1N


[druglikeness]: [Mean: 0.478   STD: 0.069   MIN: 0.000   MAX: 0.690]
********************************************************************************


GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs


Load Pre-trained Discriminator.
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs


Adversarial Training...



Epoch 1 / 100, G_STEP 1 / 1, PG_Loss: -1.757
Generating 5000 samples...
100%|███████████████████████████████████████████████████████████████████| 78/78 [00:14<00:00,  5.41it/s]

Total Computational Time:  0.22  hours.

Results Report:
********************************************************************************
Total Mols:   4992
Validity:     4499    (90.12%)
Uniqueness:   4185    (93.02%)
Novelty:      3494    (83.49%)
Diversity:    0.92


Samples of Novel SMILES:
CC12CC(O)(C1)[C]([NH])O2
CCC(=NC)N(C)C#N
C#CC1CC12CC(=O)C2
OC12CCCOC1C2
C1=CC2C3CC1C1C2C31


[druglikeness]: [Mean: 0.477   STD: 0.071   MIN: 0.162   MAX: 0.658]
********************************************************************************


:
:
:


LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]



Epoch 99 / 100, G_STEP 1 / 1, PG_Loss: -3.988
Generating 5000 samples...
100%|███████████████████████████████████████████████████████████████████| 78/78 [00:26<00:00,  3.00it/s]

Total Computational Time:  3.94  hours.

Results Report:
********************************************************************************
Total Mols:   4992
Validity:     4914    (98.44%)
Uniqueness:   2882    (58.65%)
Novelty:      2757    (95.66%)
Diversity:    0.90


Samples of Novel SMILES:
CCCCC(N)CNC
CCOCCC(C)C
CC(N)CC(C)CC#N
CCC(C)(CC)CO
COCC(C)OCC(C)O


[druglikeness]: [Mean: 0.555   STD: 0.055   MIN: 0.247   MAX: 0.673]
********************************************************************************


LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]



Epoch 100 / 100, G_STEP 1 / 1, PG_Loss: 0.952
Generating 5000 samples...
100%|███████████████████████████████████████████████████████████████████| 78/78 [00:24<00:00,  3.20it/s]

Total Computational Time:  4.00  hours.

Results Report:
********************************************************************************
Total Mols:   4992
Validity:     4891    (97.98%)
Uniqueness:   2826    (57.78%)
Novelty:      2689    (95.15%)
Diversity:    0.90


Samples of Novel SMILES:
CCC(=N)NC(C)=O
CC(C)C(C)CCCCO
CC(C=O)C1(C)CCO1
CCOCCC(C)C
CCC(COC)C(C)N


[druglikeness]: [Mean: 0.555   STD: 0.055   MIN: 0.311   MAX: 0.687]
********************************************************************************


LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Top-12 Molecules of [druglikeness]:
c1[nH]c(CCOC)nc1C        0.687
C(C)c1c(N)onc1OC         0.674
[nH]1ncc(CCC)c1NC        0.671
Nc1nc(CCC)ncc1   0.671
c1(OCCC)ncc[nH]1         0.668
CCCc1cc(N)oc1    0.659
o1c(CCCO)ncc1    0.658
n1c(OCCC)n[nH]c1         0.658
C1CCC(CCC(=O)O)C1        0.657
CC(C)C(O)CC(OC)C         0.656
CCc1c(NC)ocn1    0.656
C(c1occ(CO)c1)C          0.655
********************************************************************************


File names for drawing distributions: ['res/generated_smiles_QM9.csv', 'res/generated_smiles_ZINC.csv']
Mean Real QED Score: 0.479
Mean GAN QED Score: 0.559
Mean WGAN QED Score: 0.836

Mean Real SA Score: 0.263
Mean GAN SA Score: 0.553
Mean WGAN SA Score: 0.886

Mean Real logP Score: 0.299
Mean GAN logP Score: 0.423
Mean WGAN logP Score: 0.643

********************************************************************************
```

# 演習内容
複数属性の最適化を行っていく。
既存のアルゴリズムでは「druglikeness」「solubility」「synthesizability」のうちどれか一つに対する最適化を行っている。

## 単純和
「druglikeness」「solubility」「synthesizability」のすべてのスコアを単純に足し合わせる。
コマンドライン引数の```--properties```オプションに```all```という設定を追加した。
これを指定することで、3つの属性すべてが最適化対象となる。

### 実装
単純和の計算用関数は以下のように実装した。
```python
def batch_all_with_weight(smiles, weight=[1/3, 1/3, 1/3]):
    val_d = batch_druglikeness(smiles)
    val_sol = batch_solubility(smiles)
    val_SA = batch_SA(smiles)
    vals = [weight[0] * val_d[i] + weight[1] * val_sol[i] + weight[2] * val_SA[i] for i in range(len(smiles))]
    return vals
```

そして、mol_metrics.pyのreward_fn()関数に処理を追加した。
```python
def reward_fn(properties, generated_smiles):
    if properties == 'druglikeness':
        vals = batch_druglikeness(generated_smiles) 
    elif properties == 'solubility':
        vals = batch_solubility(generated_smiles)
    elif properties == 'synthesizability':
        vals = batch_SA(generated_smiles)
    # 2025/05/26 allオプションの追加
    elif properties == 'all':
        vals = batch_all_with_weight(generated_smiles)
    return vals
```

utils.pyのtop_mols_show()関数にも```all```分岐での処理を追加した。
```python
def top_mols_show(filename, properties):
    """
		filename: NEGATIVE FILES (generated dataset of SMILES)
		properties: 'druglikeness' or 'solubility' or 'synthesizability'
    """
    mols, scores = [], []
    # Read the generated SMILES data
    smiles = open(filename, 'r').read()
    smiles = list(smiles.split('\n'))  
        
    if properties == 'druglikeness':
        scores = batch_druglikeness(smiles)      
    elif properties == 'synthesizability':
        scores = batch_SA(smiles)  
    elif properties == 'solubility':
        scores = batch_solubility(smiles)
	elif properties == 'all':
		scores = batch_all_with_weight(smiles)

  	# Sort the scores
    dic = dict(zip(smiles, scores))
    dic=sorted(dic.items(),key=lambda x:x[1],reverse=True)
:
:
```

### 結果
```
Total Computational Time: 3.36 hours.

Results Report:
********************************************************************************
Total Mols:   4992
Validity:     4937    (98.90%)
Uniqueness:   613    (12.42%)
Novelty:      595    (97.06%)
Diversity:    0.86


Samples of Novel SMILES:
CC(C)CC1(C)CCC1C
CCCCC(C)OCCC
CCCC(C)(C=O)CC
CCCCc1cc(C)n[nH]1
CCCOCC(C)CO


[all]: [Mean: 0.601   STD: 0.067   MIN: 0.341   MAX: 0.749]
********************************************************************************


Top-12 Molecules of [all]:
C(C)CCCCCCCCC 	 0.749
C(CCCCCCCC)(C)C 	 0.747
C(C)CCCCCCC(C)C 	 0.747
C(CCCCCC)CCC 	 0.739
C(C)CCCCCCCC 	 0.739
C(CCCCCCC)CC 	 0.739
C(CCC)CCCCC(C) 	 0.739
C(CCCCCCCC)C 	 0.739
C(CCCCC)CCCC 	 0.739
C(CCCCCCCC)(C) 	 0.739
C(CCC)CCCCCC 	 0.739
C(CC)CCCCCCC 	 0.739
********************************************************************************


File names for drawing distributions: ['res/generated_smiles_QM9.csv', 'res/generated_smiles_ZINC.csv']
Mean Real QED Score: 0.479
Mean GAN QED Score: 0.531
Mean WGAN QED Score: 0.777

Mean Real SA Score: 0.263
Mean GAN SA Score: 0.725
Mean WGAN SA Score: 0.868

Mean Real logP Score: 0.299
Mean GAN logP Score: 0.634
Mean WGAN logP Score: 0.673

********************************************************************************
```

### 重み変更($\frac{2}{3} \frac{1}{6} \frac{1}{6}$)
```
Epoch 100 / 100, G_STEP 1 / 1, PG_Loss: -2.090
Generating 5000 samples...
100%|███████████████████████████████████████████████████████████████████| 78/78 [00:12<00:00,  6.34it/s]

Total Computational Time:  3.77  hours.

Results Report:
********************************************************************************
Total Mols:   4992
Validity:     4930    (98.76%)
Uniqueness:   1659    (33.65%)
Novelty:      1636    (98.61%)
Diversity:    0.86


Samples of Novel SMILES:
CCC(O)C(O)CC(C)O
CC(C)CCOC(C)O
CCC(O)CC(C)C(C)C
CCOCCC(O)CCO
CCC(C)C(C=O)CC


[all]: [Mean: 0.572   STD: 0.040   MIN: 0.338   MAX: 0.659]
********************************************************************************


LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Top-12 Molecules of [all]:
C(CCCCCCCC)C     0.739
C(CCCCC)CCCC     0.739
C(CCCCCC)C(C)C   0.738
C(CCCCCCC)(C)C   0.738
C(CCC)CCCCC      0.728
C(C)CCCCCCC      0.728
C(CCC)CCC(C)C    0.727
C(C)(C)CCCCCC    0.727
C(CCCCC)C(C)C    0.727
C(CCCCCC)(C)C    0.727
CCCCCCC(C)C      0.727
C(CCC(C)C)CCC    0.727
********************************************************************************


File names for drawing distributions: ['res/generated_smiles_QM9.csv', 'res/generated_smiles_ZINC.csv']
Mean Real QED Score: 0.479
Mean GAN QED Score: 0.575
Mean WGAN QED Score: 0.777

Mean Real SA Score: 0.263
Mean GAN SA Score: 0.616
Mean WGAN SA Score: 0.868

Mean Real logP Score: 0.299
Mean GAN logP Score: 0.560
Mean WGAN logP Score: 0.673

********************************************************************************
```

### 重み($\frac{2}{3} \frac{1}{6} \frac{1}{6}$) + WGAN + minibatch
```
Epoch 100 / 100, G_STEP 1 / 1, PG_Loss: -4.141
Generating 5000 samples...
100%|███████████████████████████████████████████████████████████████████| 78/78 [00:10<00:00,  7.27it/s]

Total Computational Time:  3.52  hours.

Results Report:
********************************************************************************
Total Mols:   4992
Validity:     4905    (98.26%)
Uniqueness:   1124    (22.92%)
Novelty:      1106    (98.40%)
Diversity:    0.88


Samples of Novel SMILES:
CCCC(N)C(C)CC
CCOC(C)CC(C)OC
CCCC(C=O)C(C)C
CCC(C)C(=O)C(C)N
CC(C)CC(C)(C)C


[all]: [Mean: 0.571   STD: 0.040   MIN: 0.384   MAX: 0.648]
********************************************************************************


LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Top-12 Molecules of [all]:
C1CCC(CCCC)CC1   0.740
C(CC)CCCCCCC     0.739
C(CCCCCCCC)C     0.739
C(CCCCCCC)CC     0.739
C(CCCC)CCCCC     0.739
CCCCCCCCCC       0.739
C(CCCCCC)C(C)C   0.738
C(CCC)CCCC(C)C   0.738
CC(CCCCCCC)C     0.738
CCCCCCCC(C)C     0.738
C(CCCCCCC)(C)C   0.738
C(C)(CCCCCCC)C   0.738
********************************************************************************


File names for drawing distributions: ['res/generated_smiles_QM9.csv', 'res/generated_smiles_ZINC.csv']
Mean Real QED Score: 0.479
Mean GAN QED Score: 0.555
Mean WGAN QED Score: 0.777

Mean Real SA Score: 0.263
Mean GAN SA Score: 0.686
Mean WGAN SA Score: 0.868

Mean Real logP Score: 0.299
Mean GAN logP Score: 0.577
Mean WGAN logP Score: 0.673

********************************************************************************
```

### 5.1.4 + LR(8e-6)
```
Total Computational Time: [1;35m 4.07 [0m hours.

Results Report:
********************************************************************************
Total Mols:   4992
Validity:     4538    (90.91%)
Uniqueness:   4404    (97.05%)
Novelty:      4139    (93.98%)
Diversity:    0.93


Samples of Novel SMILES:
CC1COC2CC1C2
CCC(=O)CC1CCC1
COC(C#N)C1COC1
CC(=O)OC(=O)NC=N
CCOC(C)C(=O)NC


[all]: [Mean: 0.421   STD: 0.073   MIN: 0.155   MAX: 0.653]
********************************************************************************


Top-12 Molecules of [all]:
C(CCC(C)(C)C)CC 	 0.700
C1CCCCCC(C)(C)C1 	 0.692
c1c(F)ccc(C)c1F 	 0.681
O(C)CCCCCC 	 0.678
C(CC)OCCCCC 	 0.675
C1C(CCC(O)=O)CC1 	 0.673
O=C(CCCC)CCC 	 0.672
C1C(CCC(C)C)C1 	 0.671
Fc1c(F)cccc1C 	 0.663
C(CCC)CNC(=O)C 	 0.659
CCCCCCO 	 0.657
CC(C(C)C)CCCC 	 0.656
********************************************************************************


File names for drawing distributions: ['res/generated_smiles_QM9.csv', 'res/generated_smiles_ZINC.csv']
Mean Real QED Score: 0.479
Mean GAN QED Score: 0.483
Mean WGAN QED Score: 0.777

Mean Real SA Score: 0.263
Mean GAN SA Score: 0.274
Mean WGAN SA Score: 0.868

Mean Real logP Score: 0.299
Mean GAN logP Score: 0.318
Mean WGAN logP Score: 0.673

********************************************************************************
```

# コードの修正
付属しているソースコードは一部、書き換えが必要な箇所が存在する。

1. TenGAN/env/env.ymlの117行目
```yml
- install==1.3.5
```
これをコメントアウトするか、削除する必要がある。

2. TenGAN/main.pyの36行目
```python
parser.add_argument('--gen_pretrain:', action='store_true', help='whether pretrain the dataset')
```
のうち```'--gen_pretrain:'```の末尾の:(コロン)が不要なため、元の行を削除するかコメントアウトして、以下のように書き換える。
```python
parser.add_argument('--gen_pretrain', action='store_true', help='whether pretrain the dataset')
```

3. TenGAN/discriminator.py
```non_mask```がゼロベクトルとなってしまった場合（```reward_fn()```関数の値がゼロベクトル）、
$masked\_encoded = encoded * [0, 0, ..., 0]$
$ave = [0, 0, ..., 0] / [0, 0, ..., 0] = [nan, nan, ..., nan]$
となり、ゼロ除算が発生して```ave```の値が欠損値となってしまう。
それを避けるために、$0 / 0 = 0$として```nan```をゼロに置き換える処理を追加した。

```python
def masked_mean(self, encoded, mask):
    """
        encoded: output of TransformerEncoder with size [batch_size, maxlength, d_model]
        mask: output of _padding_mask with size [maxlength, batch_size]: if pad: True, else False
        return: mean of the encoded according to the non-zero/True mask [batch_size, d_model]
    """
    non_mask = mask.transpose(0,1).unsqueeze(-1) == False # [batch_size, maxlength, 1] if Pad: 0, else 1
    masked_encoded = encoded * non_mask # [batch_size, maxlength, d_model]
    # 2025/06/02 nanが含まれる問題を解決
    # この時点でのゼロ除算は許容
    ave = masked_encoded.sum(dim=1) / non_mask.sum(dim=1) # [batch_size, d_model]
    # NaNがある場合、それをゼロベクトルで置換
    ave = torch.where(torch.isnan(ave), torch.zeros_like(ave), ave)

    return ave
```

# Author
大阪大学大学院 情報科学研究科 マルチメディア工学専攻 データ生成工学講座 1年
松村優大
