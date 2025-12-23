#!/usr/bin/env python3
"""
TenGAN Property Distribution Plotter (Figure 4 Style)

TenGAN論文のFigure 4のようなプロパティ分布グラフを作成するプログラム。
Validity, Uniqueness, Noveltyをすべて満たした分子のみを使用。
QED, SA, logPをそれぞれ別ファイルに出力。
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys
import os
from scipy.stats import gaussian_kde

# TenGANの訓練プログラムで使用されている関数をインポート
# mol_metrics.pyがSA_score.pkl.gzを読み込むため、TenGAN/ディレクトリに移動してからインポート
original_dir = os.getcwd()
sys.path.insert(0, os.path.join(original_dir, 'TenGAN'))
os.chdir(os.path.join(original_dir, 'TenGAN'))
from mol_metrics import batch_druglikeness, batch_solubility, batch_SA
os.chdir(original_dir)


def calculate_properties(smiles):
    """
    SMILESからQED, SA, logPを計算
    TenGAN/mol_metrics.pyの関数を直接使用

    Returns:
        tuple: (valid, qed, sa, logp) valid=1なら有効な分子
    """
    try:
        # TenGAN/mol_metrics.pyの関数を使用
        qed_list = batch_druglikeness([smiles])
        logp_list = batch_solubility([smiles])
        sa_list = batch_SA([smiles])

        qed = qed_list[0]
        logp = logp_list[0]
        sa = sa_list[0]

        # すべて0.0の場合は無効とみなす
        # if qed == 0.0 and logp == 0.0 and sa == 0.0:
        #     return (0, None, None, None)

        return (1, qed, sa, logp)
    except:
        return (0, None, None, None)


def load_original_dataset(csv_path):
    """
    オリジナルデータセット (SMILESのみ) を読み込み、プロパティを計算
    すべての分子を使用 (計算失敗した分子も0.0として含める)

    Returns:
        dict: {'QED': [...], 'SA': [...], 'logP': [...]}
    """
    data = {'QED': [], 'SA': [], 'logP': []}
    total_count = 0

    print(f"オリジナルデータセットを読み込み中: {csv_path}")

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            smiles = row[0].strip()
            total_count += 1

            # プロパティ計算 (すべての分子を使用)
            valid, qed, sa, logp = calculate_properties(smiles)

            # 計算失敗した場合も0.0として追加
            data['QED'].append(qed if valid else 0.0)
            data['SA'].append(sa if valid else 0.0)
            data['logP'].append(logp if valid else 0.0)

    print(f"  総分子数: {total_count}")
    return data


def load_generated_csv(csv_path):
    """
    生成された分子のCSVを読み込み (valid=1, unique=1, novel=1のみ)

    Returns:
        dict: {'QED': [...], 'SA': [...], 'logP': [...]}
    """
    data = {'QED': [], 'SA': [], 'logP': []}

    print(f"生成データを読み込み中: {csv_path}")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        total = 0
        filtered = 0

        for row in reader:
            total += 1

            # valid, unique, novelがすべて1かチェック
            if (row.get('valid') == '1' and
                row.get('unique') == '1' and
                row.get('novel') == '1'):

                try:
                    qed = float(row['QED']) if row['QED'] else None
                    sa = float(row['SA']) if row['SA'] else None
                    logp = float(row['logP']) if row['logP'] else None

                    if qed is not None:
                        data['QED'].append(qed)
                    if sa is not None:
                        data['SA'].append(sa)
                    if logp is not None:
                        data['logP'].append(logp)

                    filtered += 1
                except (ValueError, KeyError):
                    continue

    print(f"  総分子数: {total}")
    print(f"  フィルタ後: {filtered} ({filtered/total*100:.2f}%)")
    print(f"  QED有効数: {len(data['QED'])}, SA有効数: {len(data['SA'])}, logP有効数: {len(data['logP'])}")

    return data


def plot_single_property(original_data, generated_data, property_name,
                         model_name, output_path, x_range=None):
    """
    1つのプロパティの分布を比較プロット (KDE使用)

    Args:
        original_data: オリジナルデータのリスト
        generated_data: 生成データのリスト
        property_name: プロパティ名 ('QED', 'SA', 'logP')
        model_name: モデル名
        output_path: 出力ファイルパス
        x_range: x軸の範囲 (min, max)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # x軸の範囲を決定
    if x_range:
        x_min, x_max = x_range
    elif property_name in ['QED', 'SA', 'logP']:
        x_min, x_max = 0.0, 1.0
    else:
        # logPなど
        all_data = []
        if original_data and len(original_data) > 0:
            all_data.extend(original_data)
        if generated_data and len(generated_data) > 0:
            all_data.extend(generated_data)
        if all_data:
            x_min = min(all_data) - 0.5
            x_max = max(all_data) + 0.5
        else:
            x_min, x_max = -2, 4

    x_grid = np.linspace(x_min, x_max, 300)

    # オリジナルデータの分布
    if original_data and len(original_data) > 0:
        mean_orig = np.mean(original_data)
        std_orig = np.std(original_data)

        # KDEで滑らかな分布を計算
        kde_orig = gaussian_kde(original_data)
        density_orig = kde_orig(x_grid)

        # 塗りつぶし
        ax.fill_between(x_grid, density_orig, alpha=0.5, color='steelblue', label='ORIGINAL')
        # 輪郭線
        ax.plot(x_grid, density_orig, color='steelblue', linewidth=2)

        print(f"  ORIGINAL - μ={mean_orig:.3f}, σ={std_orig:.3f}")

    # 生成データの分布
    stats_text_lines = []
    if generated_data and len(generated_data) > 0:
        mean_gen = np.mean(generated_data)
        std_gen = np.std(generated_data)

        # KDEで滑らかな分布を計算
        kde_gen = gaussian_kde(generated_data)
        density_gen = kde_gen(x_grid)

        # 塗りつぶし
        ax.fill_between(x_grid, density_gen, alpha=0.5, color='orange', label=model_name)
        # 輪郭線
        ax.plot(x_grid, density_gen, color='orange', linewidth=2)

        print(f"  {model_name} - μ={mean_gen:.3f}, σ={std_gen:.3f}")

        # 統計情報テキスト (両方の情報を表示)
        # if original_data and len(original_data) > 0:
        #     stats_text_lines.append(f'ORIGINAL: μ = {mean_orig:.2f}, σ = {std_orig:.2f}')
        # stats_text_lines.append(f'{model_name}: μ = {mean_gen:.2f}, σ = {std_gen:.2f}')

    # 統計情報をテキストで表示
    if stats_text_lines:
        stats_text = '\n'.join(stats_text_lines)
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ラベルとタイトル
    property_labels = {
        'QED': ('QED Score', 'Drug-likeness (QED)'),
        'SA': ('SA Score', 'Synthesizability (SA)'),
        'logP': ('logP Score', 'Solubility (logP)')
    }

    xlabel, title = property_labels.get(property_name, (property_name, property_name))

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(f'{title} Distribution', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # x軸の範囲を設定
    ax.set_xlim([x_min, x_max])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    print(f"保存: {output_path}")
    plt.close()


def print_statistics(original_data, generated_data, model_name):
    """統計情報を表示"""
    print("\n" + "="*80)
    print(f"統計情報比較: ORIGINAL vs {model_name}")
    print("="*80)

    for prop in ['QED', 'SA', 'logP']:
        print(f"\n{prop}:")

        if prop in original_data and original_data[prop]:
            orig = original_data[prop]
            print(f"  ORIGINAL:")
            print(f"    分子数: {len(orig)}")
            print(f"    平均: {np.mean(orig):.4f}")
            print(f"    標準偏差: {np.std(orig):.4f}")
            print(f"    最小値: {np.min(orig):.4f}")
            print(f"    最大値: {np.max(orig):.4f}")

        if prop in generated_data and generated_data[prop]:
            gen = generated_data[prop]
            print(f"  {model_name}:")
            print(f"    分子数: {len(gen)}")
            print(f"    平均: {np.mean(gen):.4f}")
            print(f"    標準偏差: {np.std(gen):.4f}")
            print(f"    最小値: {np.min(gen):.4f}")
            print(f"    最大値: {np.max(gen):.4f}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='TenGAN プロパティ分布プロッター (Figure 4スタイル)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('csv_file', type=str,
                       help='生成分子の評価結果CSVファイルパス')
    parser.add_argument('--original', '-orig', type=str,
                       default='TenGAN/dataset/QM9.csv',
                       help='オリジナルデータセットのCSVファイル (デフォルト: TenGAN/dataset/QM9.csv)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='出力先ディレクトリ (デフォルト: CSVと同じディレクトリ)')
    parser.add_argument('--model-name', '-m', type=str, default='ours',
                       help='モデル名 (デフォルト: ours)')

    args = parser.parse_args()

    # CSVファイルの確認
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"エラー: CSVファイルが見つかりません: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    # 生成データの読み込み
    generated_data = load_generated_csv(str(csv_path))

    # オリジナルデータの読み込み
    original_path = Path(args.original)
    if not original_path.exists():
        print(f"警告: オリジナルデータが見つかりません: {args.original}")
        print("オリジナルデータなしで続行します。")
        original_data = {}
    else:
        original_data = load_original_dataset(str(original_path))

    # 統計情報表示
    print_statistics(original_data, generated_data, args.model_name)

    # 出力ディレクトリ
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = csv_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # 各プロパティごとにグラフを生成
    base_name = csv_path.stem

    print("\nグラフを生成中...")

    # QED分布
    print("\nQED分布:")
    output_path = output_dir / f"{base_name}_QED_distribution.png"
    plot_single_property(
        original_data.get('QED', []),
        generated_data.get('QED', []),
        'QED',
        args.model_name,
        str(output_path),
        x_range=(0.0, 1.0)
    )

    # SA分布
    print("\nSA分布:")
    output_path = output_dir / f"{base_name}_SA_distribution.png"
    plot_single_property(
        original_data.get('SA', []),
        generated_data.get('SA', []),
        'SA',
        args.model_name,
        str(output_path),
        x_range=(0.0, 1.0)
    )

    # logP分布
    print("\nlogP分布:")
    output_path = output_dir / f"{base_name}_logP_distribution.png"
    plot_single_property(
        original_data.get('logP', []),
        generated_data.get('logP', []),
        'logP',
        args.model_name,
        str(output_path),
        x_range=None  # logPは範囲を自動設定
    )

    print(f"\nすべてのグラフを保存しました: {output_dir}")
    print("完了!")


if __name__ == '__main__':
    main()
