import pandas as pd
import os
import glob


# tables full vocab:
input_dir = '../output/tables/'
outputfile = 'latex_tables.txt'
with open(input_dir+'/'+outputfile, 'w') as f:
    for filepath in sorted(glob.glob(input_dir+'/*.xlsx')):
        df = pd.read_excel(filepath)
        df = df.fillna('-')
        #df = df.set_index(df.columns[0])
        df.index.name = 'feature type'
        f.write('\n\n\\begin{table}[H]\\label{'+os.path.basename(filepath)+'}\n\\begin{tiny}')
        f.write(df.to_latex(bold_rows=True))
        f.write('\n\\end{tiny}\\caption{'+os.path.basename(filepath).replace('_', ' ')+\
            '}\n\\end{table}\n\n')

"""
# tables verif signif:
input_dir = 'signif_inst_word1'
outputfile = 'latex_tables.txt'
with open(input_dir+'/'+outputfile, 'w') as f:
    for filepath in sorted(glob.glob(input_dir+'/*.csv')):
        df = pd.read_csv(filepath)
        df = df.set_index(df.columns[0])
        df = df.fillna('-')
        df.index.name = ''
        f.write('\n\n\\begin{table}[H]\\label{'+os.path.basename(filepath)+'}\n\\begin{tiny}')
        f.write(df.to_latex(bold_rows=True))
        f.write('\n\\end{tiny}\\caption{'+os.path.basename(filepath).replace('_', ' ')+\
            '}\n\\end{table}\n\n')
"""


