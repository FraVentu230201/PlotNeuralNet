import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pycore.tikzeng import *
from pycore.blocks import block_Res

# descrizione dell’architettura
arch = [
    to_head('../'),                 # include i file .sty
    to_cor(),
    to_begin(),

    to_input('../examples/fcn8s/cats.jpg', width=8, height=8, name='input'),

    to_Conv('conv1', 64, 64, offset="(1,0,0)", to="(0,0,0)",
            height=64, depth=64, width=2),

    to_Pool('pool1', offset="(0,0,0)", to="(conv1-east)",
            height=32, depth=32, width=1),

    *block_Res(
        num=3, name='res2',
        botton='pool1', top='res2_3',
        s_filer=64, n_filer=64,
        offset="(1,0,0)", size=(32,32,2)
    ),

    to_SoftMax('softmax', 1000, "(3,0,0)", "(res2_3-east)", caption="Softmax"),

    to_end()
]

def main():
    namefile = sys.argv[0].split('.')[0]   # "resnet"
    to_generate(arch, namefile + '.tex')   # crea resnet.tex

if __name__ == '__main__':
    main()
