import os

# Commit comment
print("===== Input The Comment About Pushing =====")
comment = input()

# Push
os.system('git commit -a -m "{}"'.format(comment))
os.system('git push origin master')