# evaluate.py - SMILES分子評価スクリプト

## 概要

`evaluate.py`は、生成されたSMILES分子を評価し、以下の指標を計算するスクリプトです：

- **Validity**: 有効なSMILES表現かどうか
- **Uniqueness**: ユニークな分子かどうか（重複排除）
- **Novelty**: 訓練データにない新規分子かどうか
- **QED**: Druglikeness（薬らしさ）スコア
- **SA**: Synthetic Accessibility（合成容易性）スコア
- **logP**: Lipophilicity（脂溶性）スコア

## 使用方法

### 基本的な使用方法

```bash
cd TenGAN
python evaluate.py res/generated_smiles_QM9/20251216_095252.csv
```

これにより、入力CSVファイルが以下のカラムを追加して**上書き**されます：
- `smiles`: 元のSMILES文字列
- `valid`: Validity (0 or 1)
- `unique`: Uniqueness (0 or 1)
- `novel`: Novelty (0 or 1)
- `QED`: QEDスコア (NaNの場合は条件未達)
- `SA`: SAスコア (NaNの場合は条件未達)
- `logP`: logP値 (NaNの場合は条件未達)

### 訓練データを指定

```bash
python evaluate.py res/generated_smiles_QM9/20251216_095252.csv --train-data dataset/QM9.csv
```

### 別ファイルに出力

```bash
python evaluate.py res/generated_smiles_QM9/20251216_095252.csv --output results_evaluated.csv
```

## 出力例

### コンソール出力

```
================================================================================
Reading input file: res/generated_smiles_QM9/20251216_095252.csv
================================================================================

Total molecules: 5000

Loading training data: dataset/QM9.csv
Training data size: 4989 unique molecules

================================================================================
Evaluating molecules...
================================================================================

Step 1/3: Checking validity...
  Processed 1000/5000 molecules...
  Processed 2000/5000 molecules...
  ...
  Valid molecules: 4852/5000 (97.04%)

Step 2/3: Checking uniqueness...
  Unique molecules: 1450/4852 (29.88% of valid)

Step 3/3: Checking novelty and calculating properties...
  Processed 1000/5000 molecules...
  ...
  Novel molecules: 1430/1450 (98.62% of unique)

================================================================================
STATISTICS (Valid + Unique + Novel molecules only)
================================================================================

Total molecules satisfying all conditions: 1430/5000 (28.60%)

Property        Mean         Std          Min          Max         
---------------------------------------------------------------
QED             0.565123     0.062341     0.314545     0.756123    
SA              0.512345     0.189234     0.100000     0.950000    
logP            1.234567     1.123456     -1.197200    4.567890    

================================================================================

Results saved to: res/generated_smiles_QM9/20251216_095252.csv

================================================================================
SUMMARY
================================================================================
Total molecules:                 5000
Valid molecules:                 4852 (97.04%)
Unique molecules:                1450 (29.00%)
Novel molecules:                 1430 (28.60%)
Valid + Unique + Novel:          1430 (28.60%)
================================================================================
```

### CSV出力例

```csv
smiles,valid,unique,novel,QED,SA,logP
C1CN=NC[C]([NH])OCC1,1,1,1,0.496501,0.1,1.02139
C1OCCCC(CN)C1,1,1,1,0.567040,0.685639,0.7618
N=C(=N)C(CCO)C,0,0,0,,,
C(C#C)C1=CCNCO1,1,1,1,0.511659,0.145456,0.4709
CC(C)CC(C)C,1,1,0,,,
```

## スコアの解釈

### QED (Quantitative Estimate of Drug-likeness)
- **範囲**: 0-1
- **高いほど良い**: 1に近いほど薬らしい分子
- **典型値**: 0.5-0.7が一般的

### SA (Synthetic Accessibility)
- **範囲**: 0-1（元のSAスコア1-10を正規化）
- **高いほど良い**: 1に近いほど合成しやすい
- **典型値**: 0.3-0.8が一般的

### logP (Lipophilicity)
- **範囲**: 実数値（通常-2 ~ 6）
- **適切な範囲**: 0-3が理想的（薬物動態的に有利）
- **高すぎると**: 水溶性が低く、吸収が悪い
- **低すぎると**: 膜透過性が低い

## スコアがNaNになる条件

以下の条件を**全て満たす**場合のみスコアが計算されます：

1. **Valid** = 1: 有効なSMILES表現
2. **Unique** = 1: ユニークな分子（重複していない）
3. **Novel** = 1: 訓練データにない新規分子

**いずれか一つでも満たさない場合、QED, SA, logPは全てNaNになります。**

## 統計出力について

コンソール出力の統計情報は、**Validity, Uniqueness, Noveltyを全て満たした分子のみ**を対象としています。

これにより、実際に有用な新規分子のみの品質を評価できます。

## オプション

```bash
python evaluate.py --help
```

```
usage: evaluate.py [-h] [--train-data TRAIN_DATA] [--output OUTPUT] input_csv

positional arguments:
  input_csv             Input CSV file containing SMILES

optional arguments:
  -h, --help            show this help message and exit
  --train-data TRAIN_DATA
                        Training data CSV for novelty calculation 
                        (default: dataset/QM9.csv)
  --output OUTPUT       Output CSV file 
                        (default: overwrite input file)
```

## 実行例

### 1. 最新の実験結果を評価

```bash
# 最新のタイムスタンプのファイルを見つける
LATEST=$(ls -t res/generated_smiles_QM9/*.csv | head -1)

# 評価を実行
python evaluate.py $LATEST
```

### 2. 複数の実験結果を一括評価

```bash
for file in res/generated_smiles_QM9/*.csv; do
    echo "Evaluating $file..."
    python evaluate.py "$file" --output "${file%.csv}_evaluated.csv"
done
```

### 3. 結果の比較

```bash
# 複数の実験結果の統計を比較
for file in res/generated_smiles_QM9/*_evaluated.csv; do
    echo "=== $file ==="
    python -c "
import pandas as pd
df = pd.read_csv('$file')
valid_novel = df[(df['valid']==1) & (df['unique']==1) & (df['novel']==1)]
print(f'Valid+Unique+Novel: {len(valid_novel)}/{len(df)} ({len(valid_novel)/len(df)*100:.2f}%)')
print(f'QED: {valid_novel[\"QED\"].mean():.4f} ± {valid_novel[\"QED\"].std():.4f}')
print(f'SA:  {valid_novel[\"SA\"].mean():.4f} ± {valid_novel[\"SA\"].std():.4f}')
print(f'logP: {valid_novel[\"logP\"].mean():.4f} ± {valid_novel[\"logP\"].std():.4f}')
"
done
```

## トラブルシューティング

### エラー: "ModuleNotFoundError: No module named 'rdkit'"

conda環境をアクティベートしてください：

```bash
conda activate tengan_env
python evaluate.py ...
```

### エラー: "FileNotFoundError: dataset/QM9.csv"

訓練データのパスを指定してください：

```bash
python evaluate.py input.csv --train-data /path/to/training/data.csv
```

### 警告: "Training data not found"

Novelty計算には訓練データが必要です。指定しない場合、全ての分子がNovelとして扱われます。

## 技術的詳細

### Validity判定
- RDKitで分子オブジェクトに変換可能
- 原子数が2個以上

### Uniqueness判定
- Canonical SMILES形式で正規化
- 重複チェック（最初の出現のみユニーク）

### Novelty判定
- 訓練データの正規化SMILES setに含まれていない

### スコア計算
- **QED**: RDKit QED.qed()を使用
- **SA**: mol_metrics.batch_SA()を使用（正規化済み）
- **logP**: RDKit Crippen.MolLogP()を使用
