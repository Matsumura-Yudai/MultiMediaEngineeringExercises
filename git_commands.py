# import os

# # Commit comment
# print("===== Input The Comment About Pushing =====")
# comment = input()
# print("===========================================")

# # Push

# os.system('git commit -a -m "{}"'.format(comment))
# os.system('git push origin master')
import os

# Commit comment
print("===== Input The Comment About Pushing =====")
comment = input()
print("===========================================")

# --- サブモジュール側で作業 ---
print("[Step 1] Working on submodule 'TenGAN'")
os.chdir('TenGAN')  # TenGANディレクトリに移動
os.system('git add .')
os.system('git commit -m "{}"'.format(comment))
os.system('git push origin main')  # サブモジュールのブランチ（mainやmasterに注意）

# --- 親リポジトリに戻る ---
print("[Step 2] Working on parent repository")
os.chdir('..')  # 親ディレクトリに戻る

# 親リポジトリ全体を add → commit → push
os.system('git add .')  # 親リポジトリのすべての変更と、サブモジュール更新もまとめてadd
os.system('git commit -m "{}"'.format(comment))
os.system('git push origin master')  # 親リポジトリのブランチ
