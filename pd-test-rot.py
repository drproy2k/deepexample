import pandas as pd
tbl = pd.DataFrame([
    ['A', 'B', 'C'],
    ['D', 'E', 'F'],
    ['G', 'H', 'I']
])
print(tbl)
print('-----------')
print(tbl.T)

#    0  1  2
# 0  A  B  C
# 1  D  E  F
# 2  G  H  I
# -----------
#    0  1  2
# 0  A  D  G
# 1  B  E  H
# 2  C  F  I