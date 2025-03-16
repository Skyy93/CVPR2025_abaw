import pandas as pd
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

print('loading...')
df1 = pd.read_pickle('ah-frame-annotation.xz')
df2 = pd.read_pickle('ah-whisperx.xz')

print('merging...')
df1['time'] = df1['frame'] * 0.040
query = '''select df1.*, df2."index" as "index2", df2.start, df2.end, df2.word, df2.cumword
from df1 left join df2 on df1.file = df2.file and df1.time >= df2.start and df1.time < df2.end
'''
df3 = pysqldf(query)
df3.to_pickle('ah-whisperx-merged.xz')
print('done')