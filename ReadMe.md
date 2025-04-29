[TenGAN: Pure Transformer Encoders Make an Efficient Discrete GAN for De Novo Molecular Generation | Chen Li, Yoshihiro Yamanishi Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, PMLR 238:361-369, 2024.]:https://proceedings.mlr.press/v238/li24d.html
[TenGAN ソースコード]:https://github.com/naruto7283/TenGAN
[WSL Anaconda導入]:https://www.salesanalytics.co.jp/datascience/datascience141/

# マルチメディア工学演習
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
- [ローカル]()
    - [環境構築]()
    - [動作確認]()


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
```
export PATH="/home/you2002724/anaconda3/bin:$PATH"
```
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
```
- install==1.3.5
```
である。これをコメントアウトするか、削除する必要がある。

また、TenGAN/main.pyの36行目の
```
parser.add_argument('--gen_pretrain:', action='store_true', help='whether pretrain the dataset')
```
のうち```'--gen_pretrain:'```の末尾の:(コロン)が不要なため、元の行を削除するかコメントアウトして、以下のように書き換える。
```
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

## プログラムの実行
[TenGAN ソースコード]に付属のmain.pyを実行してみた。
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
データセットQM9.csvを用いて実行したいと思ったが、失敗した。
```
(tengan_env) you2002724@UD0724:~/workspace/MultiMediaEngineeringExercises/TenGAN$ python main.py --gen_pretrain --dataset_name QM9 --max_len 60



Vocabulary Information:
==================================================================
{' ': 0, '^': 1, '$': 2, 'H': 3, 'B': 4, 'c': 5, 'C': 6, 'n': 7, 'N': 8, 'o': 9, 'O': 10, 'p': 11, 'P': 12, 's': 13, 'S': 14, 'F': 15, 'Q': 16, 'W': 17, 'I': 18, '[': 19, ']': 20, '+': 21, 'Z': 22, 'X': 23, '-': 24, '=': 25, '#': 26, '.': 27, '(': 28, ')': 29, '1': 30, '2': 31, '3': 32, '4': 33, '5': 34, '6': 35, '7': 36, '@': 37, '/': 38, '\\': 39}


Parameter Information:
==================================================================
POSITIVE_FILE            :   dataset/QM9.csv
NEGATIVE_FILE            :   res/generated_smiles_QM9.csv
G_PRETRAINED_MODEL       :   res/save_models/QM9/TenGAN_0.5/rollout_8//batch_64/druglikeness/g_pretrained.pkl
D_PRETRAINED_MODEL       :   res/save_models/QM9/TenGAN_0.5/rollout_8//batch_64/druglikeness/d_pretrained.pkl
PROPERTY_FILE            :   res/save_models/QM9/TenGAN_0.5/rollout_8//batch_64/druglikeness/trained_results.csv
BATCH_SIZE               :   64
MAX_LEN                  :   60
VOCAB_SIZE               :   40
DEVICE                   :   cuda
GPUS                     :   1


GEN_PRETRAIN             :   True
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



Start time is 2025-04-27 14:54:40



GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs


Pre-train Generator...
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Epoch 149: 100%|███████████████████████| 79/79 [00:02<00:00, 35.31it/s, loss=0.534, v_num=0, val_loss=0.891]
Generator Pre-train Time: 0.28 hours
Generating 10000 samples...
100%|█████████████████████████████████████████████████████████████████████| 156/156 [00:29<00:00,  5.34it/s]

Results Report:
********************************************************************************
Total Mols:   9984
Validity:     8978    (89.92%)
Uniqueness:   7961    (88.67%)
Novelty:      6758    (84.89%)
Diversity:    0.92


Samples of Novel SMILES:
CCC1C(=O)C12COC2
N#CCCCC1CCC1
CC12C[C]1[CH]C1NC12
CCC(CC)CC(C)C
C#CC1CC2(O)C(O)C12


[druglikeness]: [Mean: 0.476   STD: 0.070   MIN: 0.126   MAX: 0.668]
********************************************************************************


GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs


Load Pre-trained Discriminator.
Traceback (most recent call last):
  File "main.py", line 482, in <module>
    main()
  File "main.py", line 360, in main
    dis.load_state_dict(torch.load(D_PRETRAINED_MODEL))
  File "/home/you2002724/anaconda3/envs/tengan_env/lib/python3.6/site-packages/torch/serialization.py", line 579, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/home/you2002724/anaconda3/envs/tengan_env/lib/python3.6/site-packages/torch/serialization.py", line 230, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/home/you2002724/anaconda3/envs/tengan_env/lib/python3.6/site-packages/torch/serialization.py", line 211, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'res/save_models/QM9/TenGAN_0.5/rollout_8//batch_64/druglikeness/d_pretrained.pkl'
```