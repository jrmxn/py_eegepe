
from pathlib import Path
import glob
import pickle
import re
import pandas as pd

d = Path('/home/mcintosh/Cloud/DataPort/2017-04-21_LIINC_go_nogo/proc_v0.0.5/')
d_match = str(d) + '/' + '*/' + 'EEG/' + '*meta.dat'
a = glob.glob(d_match)
df = pd.DataFrame([], columns=['sub', 'run', 'f'])
for f_meta in a:
    with open(str(f_meta), 'rb') as handle:
        meta = pickle.load(handle)
        str_sub = Path(f_meta).parts[-3]
        p = re.compile(r'.+\((\w+)\)')
        str_sub = p.findall(str_sub)[0]

        str_run = Path(f_meta).parts[-1]
        p = re.compile(r'.+-(\d+).+')
        int_run = int(p.findall(str_run)[0])

        f_alpha = meta['f_alpha']

        df = df.append({'sub':str_sub, 'run':int_run, 'f':f_alpha}, ignore_index=True)
        s_out = str_sub + '  ' + str(int_run) + '  ' + str(f_alpha)
        print(s_out)

# print(df)

dm = df.mean()
dmm = df.groupby('sub').mean().mean()
dms = df.groupby('sub').mean().std()
dsm = df.groupby('sub').std().mean()
s = ''
s = s + f'The average frequency is {dm}'
s = s + f'The average of the average across run frequency is {dmm}'
s = s + f'The average of the std across run frequency is {dsm}'
s = s + f'The std of the average across run frequency is {dms}'
print(s)

