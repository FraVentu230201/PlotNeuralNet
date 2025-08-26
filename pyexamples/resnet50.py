import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pycore.tikzeng import *
from pycore.blocks import block_Res

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    #importante: dopo che si fa una modifica per poter vedere se è andata a buon fine bisogna salvare!!

    # (opzionale) immagine di input
    # to_input('../examples/fcn8s/cats.jpg', width=6, height=6, name='img'),

    # conv1 7x7 s=2 + maxpool 3x3 s=2  (niente virgole nelle caption!)
    to_Conv('conv1', 64, 112, offset="(1,0,0)", to="(0,0,0)",
        height=64, depth=64, width=2, caption='{${7\\times7}$, s=2}'),
    to_Pool('pool1', offset="(0,0,0)", to="(conv1-east)",
        height=32, depth=32, width=1, caption='{${3\\times3}$, s=2}'),

    # ResNet-50: bottleneck (3, 4, 6, 3)
    *block_Res(num=3, name='conv2', botton='pool1',  top='conv2_3',
               s_filer=56, n_filer=256,  offset="(1,0,0)", size=(32,32,2)),
    *block_Res(num=4, name='conv3', botton='conv2_3', top='conv3_4',
               s_filer=28, n_filer=512,  offset="(1,0,0)", size=(28,28,2)),
    *block_Res(num=6, name='conv4', botton='conv3_4', top='conv4_6',
               s_filer=14, n_filer=1024, offset="(1,0,0)", size=(24,24,2)),
    *block_Res(num=3, name='conv5', botton='conv4_6', top='conv5_3',
               s_filer=7,  n_filer=2048, offset="(1,0,0)", size=(16,16,2)),

    # GAP + FC
    to_Pool('avgpool', offset="(1,0,0)", to="(conv5_3-east)",
        height=6, depth=6, width=1, caption='{\\footnotesize avgpool}'),
    to_SoftMax('fc', 1000, offset="(1,0,0)", to="(avgpool-east)",
           width=1, height=1, depth=1, caption='{\\footnotesize 1000}'),


    to_end()
]

def main():
    namefile = os.path.splitext(os.path.basename(__file__))[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
