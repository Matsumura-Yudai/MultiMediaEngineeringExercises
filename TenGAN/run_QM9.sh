#!/bin/bash
set -e  # シェルスクリプト内で何らかのエラーが発生した時点で、シェルスクリプトを終了
set -u  # 未定義の変数を参照しようとした際にシェルスクリプトをエラー終了
# set -x  # 実行したコマンドをすべて標準エラー出力
# set -C  # > で出力のリダイレクト先に既存のファイルを指定すると、元々の中身を上書きしてしまうが、-C オプションを使うと上書きしようとした際にエラーにしてくれる
# set -o pipefail # パイプラインの途中でエラーが起きた時に、パイラプイン全体の終了ステータスをエラーの起きたコマンドの終了ステータスと同じにしてくれる

# ===========================
# Command Line
# ===========================
GEN_FLAG=$1
DIS_FLAG=$2
ADV_FLAG=$3
LOG_FLAG=$4

# ===========================
# Log Files
# ===========================
LOG_STD="$HOME/workspace/MultiMediaEngineeringExercises/TenGAN/log_files/std.log"
LOG_ERR="$HOME/workspace/MultiMediaEngineeringExercises/TenGAN/log_files/stderr.log"

# ===========================
# General Hyperparameters
# ===========================   # default
DATASET_NAME="QM9"              # "ZINC"
MAX_LEN=60                      # 60
BATCH_SIZE=64                   # 64

# ===========================
# Generator Parameters
# ===========================
GENERATED_NUM=5000              # 5000
GEN_TRAIN_SIZE=4800             # 4800
GEN_NUM_ENCODER_LAYERS=4        # 4
GEN_D_MODEL=128                  # 128
GEN_DIM_FEEDFORWARD=1024        # 1024
GEN_NUM_HEADS=4                 # 4
GEN_MAX_LR=8e-4                 # 8e-4
GEN_DROPOUT=0.1                 # 0.1
GEN_EPOCHS=100                  # 100

# ===========================
# Discriminator Parameters
# ===========================
DIS_NUM_ENCODER_LAYERS=4        # 4
DIS_D_MODEL=128                  # 128
DIS_NUM_HEADS=4                 # 4
DIS_EPOCHS=10                   # 10
DIS_FEED_FORWARD=200            # 200
DIS_DROPOUT=0.25                # 0.25

# ===========================
# Adversarial Training Parameters
# ===========================
UPDATE_RATE=0.8                 # 0.8
PROPERTIES="all"       # "druglikeness"
DIS_LAMBDA=0.5                  # 0.5
ADV_LR=8e-6                     # 8e-5
SAVE_NAME=20250610                  # 66
ROLL_NUM=16                     # 16
ADV_EPOCHS=100                  # 100

# ===========================
# Reinforcement Learning
# ===========================
WEIGHTS=(2/3 1/6 1/6)

# ===========================
# Usage
# ===========================
if [ "$GEN_FLAG" != "y" ] && [ "$GEN_FLAG" != "n" ] || [ "$DIS_FLAG" != "y" ] && [ "$DIS_FLAG" != "n" ] || [ "$ADV_FLAG" != "y" ] && [ "$ADV_FLAG" != "n" ]; then
    echo "Usage: $0 (y or n) (y or n) (y or n)"
    exit 1
fi

# ===========================
# Execute
# ===========================
# conda activate tengan_env

python3 $HOME/workspace/MultiMediaEngineeringExercises/TenGAN/main.py \
--dataset_name $DATASET_NAME \
--max_len $MAX_LEN \
--batch_size $BATCH_SIZE \
\
$([[ "$GEN_FLAG" = "y" ]] && echo "--gen_pretrain") \
--generated_num $GENERATED_NUM \
--gen_train_size $GEN_TRAIN_SIZE \
--gen_num_encoder_layers $GEN_NUM_ENCODER_LAYERS \
--gen_d_model $GEN_D_MODEL \
--gen_dim_feedforward $GEN_DIM_FEEDFORWARD \
--gen_num_heads $GEN_NUM_HEADS \
--gen_max_lr $GEN_MAX_LR \
--gen_dropout $GEN_DROPOUT \
--gen_epochs $GEN_EPOCHS \
\
$([[ "$DIS_FLAG" = "y" ]] && echo "--dis_pretrain") \
--dis_wgan \
--dis_minibatch \
--dis_num_encoder_layers $DIS_NUM_ENCODER_LAYERS \
--dis_d_model $DIS_D_MODEL \
--dis_num_heads $DIS_NUM_HEADS \
--dis_epochs $DIS_EPOCHS \
--dis_feed_forward $DIS_FEED_FORWARD \
--dis_dropout $DIS_DROPOUT \
\
$([[ "$ADV_FLAG" = "y" ]] && echo "--adversarial_train") \
--update_rate $UPDATE_RATE \
--properties $PROPERTIES \
--dis_lambda $DIS_LAMBDA \
--adv_lr $ADV_LR \
--save_name $SAVE_NAME \
--roll_num $ROLL_NUM \
--adv_epochs $ADV_EPOCHS \
\
--weights "${WEIGHTS[@]}" \
# > "$LOG_STD" 2> "$LOG_ERR"