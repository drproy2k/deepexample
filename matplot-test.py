#############################################################
# matplot-test.py
# 윈도 PC에서 anaconda 사용시 아래와 같은 에러가 발생되어 실행되지 않는다.
# FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\Genie\\.matplotlib\\fontlist-v300.json'
#
from matplotlib import pyplot as plt

plt.plot([1,2,3], [110,130,120])
plt.show()

