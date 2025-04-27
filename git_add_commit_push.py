import os
from datetime import datetime

# Commit comment
now = datetime.now()
comment = "update : " + datetime.now().strftime("%Y/%m/%d-%H:%M:%S")

# Push
os.system('git add .')
os.system('git commit -m "{}"'.format(comment))
os.system('git push https://github.com/Matsumura-Yudai/graduation_work.git main')