import glob
import os
import codecs
import random
import shutil

random.seed(10001)

if not os.path.isdir('test'):
        os.mkdir('test')
authors = {}
for filename in glob.glob('caesar_test/*.txt'):
    print(filename)
    author, title, idx = os.path.basename(filename).replace('.txt', '').split('_')
    if not os.path.isdir('test/'+author):
        os.mkdir('test/'+author)
    shutil.copy(filename, 'test/'+author+'/'+title+'_'+idx+'.txt')

