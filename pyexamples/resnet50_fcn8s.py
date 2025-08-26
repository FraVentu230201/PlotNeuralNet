# pyexamples/resnet50_fcn8s.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pycore.tikzeng import *

arch = [
    # header + macro/colori standard
    to_head('..'),
    to_cor(),
    # colore extra per i deconvolution/upsample (come nel tuo LaTeX)
    r'\def\DcnvColor{rgb:blue,5;green,2.5;white,5}',
    to_begin(),

    # immagine di input (stessa dei tuoi esempi)
    to_input('../examples/fcn8s/cats.jpg', width=8, height=8, name='temp'),

    # conv1 + pool1
    r'\pic[shift={(0,0,0)}] at (0,0,0) {RightBandedBox={name=cr1,caption=conv1,'
    r'xlabel={{"64","64"}},zlabel=I,fill=\ConvColor,bandfill=\ConvReluColor,'
    r'height=40,width={2,2},depth=40}};',
    r'\pic[shift={(0,0,0)}] at (cr1-east) {Box={name=p1,fill=\PoolColor,opacity=0.5,height=35,width=1,depth=35}};',

    # conv2 + pool2
    r'\pic[shift={(2,0,0)}] at (p1-east) {RightBandedBox={name=cr2,caption=conv2,'
    r'xlabel={{"64","64"}},zlabel=I/2,fill=\ConvColor,bandfill=\ConvReluColor,'
    r'height=35,width={3,3},depth=35}};',
    r'\pic[shift={(0,0,0)}] at (cr2-east) {Box={name=p2,fill=\PoolColor,opacity=0.5,height=30,width=1,depth=30}};',

    # conv3 + pool3
    r'\pic[shift={(2,0,0)}] at (p2-east) {RightBandedBox={name=cr3,caption=conv3,'
    r'xlabel={{"256","256","256"}},zlabel=I/4,fill=\ConvColor,bandfill=\ConvReluColor,'
    r'height=30,width={4,4,4},depth=30}};',
    r'\pic[shift={(0,0,0)}] at (cr3-east) {Box={name=p3,fill=\PoolColor,opacity=0.5,height=23,width=1,depth=23}};',

    # conv4 + pool4
    r'\pic[shift={(1.8,0,0)}] at (p3-east) {RightBandedBox={name=cr4,caption=conv4,'
    r'xlabel={{"512","512","512"}},zlabel=I/8,fill=\ConvColor,bandfill=\ConvReluColor,'
    r'height=23,width={7,7,7},depth=23}};',
    r'\pic[shift={(0,0,0)}] at (cr4-east) {Box={name=p4,fill=\PoolColor,opacity=0.5,height=15,width=1,depth=15}};',

    # conv5 + pool5
    r'\pic[shift={(1.5,0,0)}] at (p4-east) {RightBandedBox={name=cr5,caption=conv5,'
    r'xlabel={{"512","512","512"}},zlabel=I/16,fill=\ConvColor,bandfill=\ConvReluColor,'
    r'height=15,width={7,7,7},depth=15}};',
    r'\pic[shift={(0,0,0)}] at (cr5-east) {Box={name=p5,fill=\PoolColor,opacity=0.5,height=10,width=1,depth=10}};',

    # fc -> conv (cr6_7) e score32
    r'\pic[shift={(1,0,0)}] at (p5-east) {RightBandedBox={name=cr6_7,caption=fc to conv,'
    r'xlabel={{"4096","4096"}},fill=\ConvColor,bandfill=\ConvReluColor,'
    r'height=10,width={10,10},depth=10}};',
    r'\pic[shift={(1,0,0)}] at (cr6_7-east) {Box={name=score32,caption=fc8 to conv,'
    r'xlabel={{"K","dummy"}},fill=\ConvColor,height=10,width=2,depth=10,zlabel=I/32}};',

    # up 32->16 + skip da p4
    r'\pic[shift={(1.5,0,0)}] at (score32-east) {Box={name=d32,xlabel={{"K","dummy"}},'
    r'fill=\DcnvColor,height=15,width=2,depth=15,zlabel=I/16}};',
    r'\pic[shift={(0,-4,0)}] at (d32-west) {Box={name=score16,xlabel={{"K","dummy"}},'
    r'fill=\ConvColor,height=15,width=2,depth=15,zlabel=I/16}};',
    r'\pic[shift={(1.5,0,0)}] at (d32-east) {Ball={name=elt1,fill=\SumColor,opacity=0.6,'
    r'radius=2.5,logo=$+$}};',

    # up 16->8 + skip da p3
    r'\pic[shift={(1.5,0,0)}] at (elt1-east) {Box={name=d16,xlabel={{"K","dummy"}},'
    r'fill=\DcnvColor,height=23,width=2,depth=23,zlabel=I/8}};',
    r'\pic[shift={(0,-6,0)}] at (d16-west) {Box={name=score8,xlabel={{"K","dummy"}},'
    r'fill=\ConvColor,height=23,width=2,depth=23,zlabel=I/8}};',
    r'\pic[shift={(1.5,0,0)}] at (d16-east) {Ball={name=elt2,fill=\SumColor,opacity=0.6,'
    r'radius=2.5,logo=$+$}};',

    # up finale + softmax
    r'\pic[shift={(2.5,0,0)}] at (elt2-east) {Box={name=d8,xlabel={{"K","dummy"}},'
    r'fill=\DcnvColor,height=40,width=2,depth=40}};',
    r'\pic[shift={(1,0,0)}] at (d8-east) {Box={name=softmax,caption=softmax,'
    r'xlabel={{"K","dummy"}},fill=\SoftmaxColor,height=40,width=2,depth=40,zlabel=I}};',

    # connessioni principali
    r'\draw [connection] (p1-east) -- node {\midarrow} (cr2-west);',
    r'\draw [connection] (p2-east) -- node {\midarrow} (cr3-west);',
    r'\draw [connection] (p3-east) -- node {\midarrow} (cr4-west);',
    r'\draw [connection] (p4-east) -- node {\midarrow} (cr5-west);',
    r'\draw [connection] (p5-east) -- node {\midarrow} (cr6_7-west);',
    r'\draw [connection] (cr6_7-east) -- node {\midarrow} (score32-west);',
    r'\draw [connection] (score32-east) -- node {\midarrow} (d32-west);',

    # skip da p4
    r'\path (p4-east) -- (cr5-west) coordinate[pos=0.25] (between4_5) ;',
    r'\draw [connection] (between4_5) -- node {\midarrow} (score16-west-|between4_5) -- node {\midarrow} (score16-west);',
    r'\draw [connection] (d32-east) -- node {\midarrow} (elt1-west);',
    r'\draw [connection] (score16-east) -- node {\midarrow} (score16-east -| elt1-south) -- node {\midarrow} (elt1-south);',
    r'\draw [connection] (elt1-east) -- node {\midarrow} (d16-west);',

    # skip da p3
    r'\path (p3-east) -- (cr4-west) coordinate[pos=0.25] (between3_4) ;',
    r'\draw [connection] (between3_4) -- node {\midarrow} (score8-west-|between3_4) -- node {\midarrow} (score8-west);',
    r'\draw [connection] (d16-east) -- node {\midarrow} (elt2-west);',
    r'\draw [connection] (score8-east) -- node {\midarrow} (score8-east -| elt2-south) -- node {\midarrow} (elt2-south);',
    r'\draw [connection] (elt2-east) -- node {\midarrow} (d8-west);',
    r'\draw [connection] (d8-east) -- node {\midarrow} (softmax-west);',

    to_end(),
]

def main():
    namefile = os.path.splitext(os.path.basename(__file__))[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
