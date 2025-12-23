#!/usr/bin/env python3
"""
evaluate.py - SMILES分子の評価スクリプト

使用方法:
    python evaluate.py <input_csv> [options]

引数:
    input_csv           : 評価するSMILESが含まれるCSVファイルパス
    --train-data        : 訓練データCSV（Novelty計算用、デフォルト: dataset/QM9.csv）
    --output            : 出力CSVファイルパス（デフォルト: 入力ファイルを上書き）

出力:
    CSVファイルに以下のカラムを追加:
    - valid: Validity (0 or 1)
    - unique: Uniqueness (0 or 1)
    - novel: Novelty (0 or 1)
    - QED: Druglikeness score (0-1, NaNの場合は条件未達)
    - SA: Synthesizability score (0-1, NaNの場合は条件未達)
    - logP: Lipophilicity score (実数値, NaNの場合は条件未達)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from mol_metrics import batch_druglikeness, batch_solubility, batch_SA

# RDKitのエラーメッセージを抑制
rdBase.DisableLog('rdApp.error')


def validate_smiles(smiles):
    """
    SMILESの妥当性を検証

    Args:
        smiles (str): SMILES文字列

    Returns:
        tuple: (is_valid, canonical_smiles, mol)
            - is_valid (bool): 有効なSMILESかどうか
            - canonical_smiles (str): 正規化されたSMILES（無効な場合はNone）
            - mol (Mol): RDKitの分子オブジェクト（無効な場合はNone）
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() <= 1:
            return False, None, None
        canonical_smiles = Chem.MolToSmiles(mol)
        return True, canonical_smiles, mol
    except:
        return False, None, None


def calculate_qed(smiles):
    """
    QED (Quantitative Estimate of Drug-likeness) を計算
    mol_metrics.pyの実装を使用

    Args:
        smiles (str): SMILES文字列

    Returns:
        float: QEDスコア (0-1)
    """
    try:
        qed_scores = batch_druglikeness([smiles])
        return qed_scores[0] if len(qed_scores) > 0 else np.nan
    except:
        return np.nan


def calculate_sa(smiles):
    """
    SA (Synthetic Accessibility) スコアを計算

    Args:
        smiles (str): SMILES文字列

    Returns:
        float: 正規化されたSAスコア (0-1, 1が最も合成しやすい)
    """
    try:
        # batch_SAは0-1に正規化されたスコアを返す（1が最も合成しやすい）
        sa_scores = batch_SA([smiles])
        return sa_scores[0] if len(sa_scores) > 0 else np.nan
    except:
        return np.nan


def calculate_logp(smiles):
    """
    logP (Solubility) を計算
    mol_metrics.pyの実装を使用（正規化済み）

    Args:
        smiles (str): SMILES文字列

    Returns:
        float: 正規化されたlogP値 (0-1)
    """
    try:
        logp_scores = batch_solubility([smiles])
        return logp_scores[0] if len(logp_scores) > 0 else np.nan
    except:
        return np.nan


def evaluate_smiles_file(input_csv, train_csv='dataset/QM9.csv', output_csv=None):
    """
    SMILESファイルを評価してスコアを追加

    Args:
        input_csv (str): 入力CSVファイルパス
        train_csv (str): 訓練データCSV（Novelty計算用）
        output_csv (str): 出力CSVファイルパス（Noneの場合は入力ファイルを上書き）
    """
    # 入力ファイルの読み込み
    print(f'\n{"="*80}')
    print(f'Reading input file: {input_csv}')
    print(f'{"="*80}\n')

    if not os.path.exists(input_csv):
        print(f'Error: Input file not found: {input_csv}')
        sys.exit(1)

    # CSVファイルを読み込み（ヘッダーなし）
    df = pd.read_csv(input_csv, names=['smiles'])
    total_count = len(df)
    print(f'Total molecules: {total_count}\n')

    # 訓練データの読み込み（Novelty計算用）
    print(f'Loading training data: {train_csv}')
    if os.path.exists(train_csv):
        train_df = pd.read_csv(train_csv, names=['smiles'])
        # 訓練データを正規化SMILES形式に変換
        train_canonical = set()
        for sm in train_df['smiles']:
            mol = Chem.MolFromSmiles(sm)
            if mol is not None:
                train_canonical.add(Chem.MolToSmiles(mol))
        print(f'Training data size: {len(train_canonical)} unique molecules\n')
    else:
        print(f'Warning: Training data not found: {train_csv}')
        print(f'Novelty will be calculated without training data (all novel)\n')
        train_canonical = set()

    # 評価の実行
    print(f'{"="*80}')
    print(f'Evaluating molecules...')
    print(f'{"="*80}\n')

    valid_list = []
    unique_list = []
    novel_list = []
    qed_list = []
    sa_list = []
    logp_list = []
    canonical_smiles_list = []

    # まず全分子のValidityとCanonical SMILESを計算
    print('Step 1/3: Checking validity...')
    for idx, smiles in enumerate(df['smiles'], 1):
        if idx % 1000 == 0:
            print(f'  Processed {idx}/{total_count} molecules...')

        is_valid, canonical_smiles, mol = validate_smiles(smiles)
        valid_list.append(1 if is_valid else 0)
        canonical_smiles_list.append(canonical_smiles)

    valid_count = sum(valid_list)
    print(f'  Valid molecules: {valid_count}/{total_count} ({valid_count/total_count*100:.2f}%)\n')

    # Uniquenessを計算（有効な分子のみ対象）
    print('Step 2/3: Checking uniqueness...')
    valid_canonical = [cs for cs, v in zip(canonical_smiles_list, valid_list) if v == 1 and cs is not None]
    unique_canonical = set()

    for cs in valid_canonical:
        if cs not in unique_canonical:
            unique_canonical.add(cs)

    # 各分子がユニークかどうかを判定
    seen = set()
    for canonical_smiles, is_valid in zip(canonical_smiles_list, valid_list):
        if not is_valid or canonical_smiles is None:
            unique_list.append(0)
        elif canonical_smiles in seen:
            unique_list.append(0)  # 重複
        else:
            unique_list.append(1)  # ユニーク
            seen.add(canonical_smiles)

    unique_count = sum(unique_list)
    print(f'  Unique molecules: {unique_count}/{valid_count} ({unique_count/valid_count*100:.2f}% of valid)\n')

    # Noveltyを計算（有効かつユニークな分子のみ対象）
    print('Step 3/3: Checking novelty and calculating properties...')
    valid_unique_novel_count = 0

    for idx, (smiles, canonical_smiles, is_valid, is_unique) in enumerate(zip(
        df['smiles'], canonical_smiles_list, valid_list, unique_list), 1):

        if idx % 1000 == 0:
            print(f'  Processed {idx}/{total_count} molecules...')

        # Validity, Uniquenessの両方を満たす場合のみNoveltyとスコアを計算
        if is_valid and is_unique and canonical_smiles is not None:
            # Novelty判定
            is_novel = canonical_smiles not in train_canonical
            novel_list.append(1 if is_novel else 0)

            # 全条件を満たす場合のみスコアを計算
            if is_novel:
                valid_unique_novel_count += 1
                # mol = Chem.MolFromSmiles(canonical_smiles)
                qed_list.append(calculate_qed(canonical_smiles))
                sa_list.append(calculate_sa(canonical_smiles))
                logp_list.append(calculate_logp(canonical_smiles))
            else:
                # Noveltyを満たさない場合はNaN
                qed_list.append(np.nan)
                sa_list.append(np.nan)
                logp_list.append(np.nan)
        else:
            # Validity or Uniquenessを満たさない場合は全てNaN
            novel_list.append(0)
            qed_list.append(np.nan)
            sa_list.append(np.nan)
            logp_list.append(np.nan)

    print(f'  Novel molecules: {valid_unique_novel_count}/{unique_count} ({valid_unique_novel_count/unique_count*100:.2f}% of unique)\n')

    # 結果をDataFrameに追加
    df['valid'] = valid_list
    df['unique'] = unique_list
    df['novel'] = novel_list
    df['QED'] = qed_list
    df['SA'] = sa_list
    df['logP'] = logp_list

    # 統計情報の計算（Validity, Uniqueness, Noveltyを全て満たす分子のみ）
    valid_scores = df[(df['valid'] == 1) & (df['unique'] == 1) & (df['novel'] == 1)]

    print(f'{"="*80}')
    print(f'STATISTICS (Valid + Unique + Novel molecules only)')
    print(f'{"="*80}\n')
    print(f'Total molecules satisfying all conditions: {len(valid_scores)}/{total_count} ({len(valid_scores)/total_count*100:.2f}%)\n')

    if len(valid_scores) > 0:
        print(f'{"Property":<15} {"Mean":<12} {"Mean(top5)":<12} {"Mean(top10)":<12} {"Mean(top100)":<12} {"Mean(top1000)":<12} {"Std":<12} {"Min":<12} {"Max":<12}')
        print(f'{"-"*111}')

        for prop in ['QED', 'SA', 'logP']:
            scores = valid_scores[prop].dropna()
            tmp = sorted(scores.tolist(), reverse=True)
            if len(scores) > 0:
                print(f'{prop:<15} {scores.mean():<12.6f} {sum(tmp[:5])/5.0:<12.6f}  {sum(tmp[:10])/10.0:<12.6f} {sum(tmp[:100])/100.0:<12.6f} {sum(tmp[:1000])/1000.0:<12.6f} {scores.std():<12.6f} {scores.min():<12.6f} {scores.max():<12.6f}')
            else:
                print(f'{prop:<15} {"N/A":<12} {"N/A":<12} {"N/A":<12} {"N/A":<12}')
    else:
        print('No molecules satisfy all conditions (Valid + Unique + Novel)')

    print(f'\n{"="*80}\n')

    # 出力ファイルの保存
    if output_csv is None:
        output_csv = input_csv

    df.to_csv(output_csv, index=False)
    print(f'Results saved to: {output_csv}\n')

    # サマリー情報
    print(f'{"="*80}')
    print(f'SUMMARY')
    print(f'{"="*80}')
    print(f'Total molecules:                 {total_count}')
    print(f'Valid molecules:                 {valid_count} ({valid_count/total_count*100:.2f}%)')
    print(f'Unique molecules:                {unique_count} ({unique_count/total_count*100:.2f}%)')
    print(f'Novel molecules:                 {valid_unique_novel_count} ({valid_unique_novel_count/total_count*100:.2f}%)')
    print(f'Valid + Unique + Novel:          {len(valid_scores)} ({len(valid_scores)/total_count*100:.2f}%)')
    print(f'{"="*80}\n')


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SMILES molecules for Validity, Uniqueness, Novelty, and molecular properties',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本的な使用方法（入力ファイルを上書き）
  python evaluate.py res/generated_smiles_QM9/20251216_095252.csv

  # 訓練データを指定
  python evaluate.py res/generated_smiles_QM9/20251216_095252.csv --train-data dataset/QM9.csv

  # 別ファイルに出力
  python evaluate.py res/generated_smiles_QM9/20251216_095252.csv --output results.csv
        """
    )

    parser.add_argument('input_csv', type=str, help='Input CSV file containing SMILES')
    parser.add_argument('--train-data', type=str, default='dataset/QM9.csv',
                        help='Training data CSV for novelty calculation (default: dataset/QM9.csv)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file (default: overwrite input file)')

    args = parser.parse_args()

    # 評価の実行
    evaluate_smiles_file(args.input_csv, args.train_data, args.output)


if __name__ == '__main__':
    main()

