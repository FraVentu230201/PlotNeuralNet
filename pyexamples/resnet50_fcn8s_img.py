# pyexamples/resnet50_fcn8s_img.py
import os, sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pycore.tikzeng import *

def input_node(img_path, width_cm=8, height_cm=8, name='temp', x=-3, y=0, z=0):
    base = os.path.dirname(__file__)
    rel = os.path.relpath(img_path, start=base).replace('\\', '/')
    return rf'\node[canvas is zy plane at x=0] ({name}) at ({x},{y},{z}) ' \
           rf'{{\includegraphics[width={width_cm}cm,height={height_cm}cm]{{\detokenize{{{rel}}}}}}};'

def build_arch(img_node):
    return [
        to_head('..'),
        to_cor(),
        r'\def\ConvColor{rgb:orange,5;red,3;white,5}',
        r'\def\ConvReluColor{rgb:orange,5;red,5;white,5}',
        r'\def\PoolColor{rgb:green,2;black,0.3}',
        r'\def\SoftmaxColor{rgb:cyan,5;black,7}',
        r'\def\SumColor{rgb:purple,5;green,10}',
        r'\def\DcnvColor{rgb:blue,2;green,4;white,5}',   # puoi sostituire anche questo
        to_begin(),

        # immagine di input scelta da te
        img_node,

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

        # fc -> conv e score32
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

        # connessioni
        r'\draw [connection] (p1-east) -- node {\midarrow} (cr2-west);',
        r'\draw [connection] (p2-east) -- node {\midarrow} (cr3-west);',
        r'\draw [connection] (p3-east) -- node {\midarrow} (cr4-west);',
        r'\draw [connection] (p4-east) -- node {\midarrow} (cr5-west);',
        r'\draw [connection] (p5-east) -- node {\midarrow} (cr6_7-west);',
        r'\draw [connection] (cr6_7-east) -- node {\midarrow} (score32-west);',
        r'\draw [connection] (score32-east) -- node {\midarrow} (d32-west);',

        r'\path (p4-east) -- (cr5-west) coordinate[pos=0.25] (between4_5) ;',
        r'\draw [connection] (between4_5) -- node {\midarrow} (score16-west-|between4_5) -- node {\midarrow} (score16-west);',
        r'\draw [connection] (d32-east) -- node {\midarrow} (elt1-west);',
        r'\draw [connection] (score16-east) -- node {\midarrow} (score16-east -| elt1-south) -- node {\midarrow} (elt1-south);',
        r'\draw [connection] (elt1-east) -- node {\midarrow} (d16-west);',

        r'\path (p3-east) -- (cr4-west) coordinate[pos=0.25] (between3_4) ;',
        r'\draw [connection] (between3_4) -- node {\midarrow} (score8-west-|between3_4) -- node {\midarrow} (score8-west);',
        r'\draw [connection] (d16-east) -- node {\midarrow} (elt2-west);',
        r'\draw [connection] (score8-east) -- node {\midarrow} (score8-east -| elt2-south) -- node {\midarrow} (elt2-south);',
        r'\draw [connection] (elt2-east) -- node {\midarrow} (d8-west);',
        r'\draw [connection] (d8-east) -- node {\midarrow} (softmax-west);',

        to_end(),
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path dell'immagine di ingresso")
    ap.add_argument("--width",  type=float, default=8.0, help="Larghezza immagine (cm)")
    ap.add_argument("--height", type=float, default=8.0, help="Altezza immagine (cm)")
    ap.add_argument("--x", type=float, default=-3.0, help="Posizione X immagine (sposta a sinistra/destra)")
    args = ap.parse_args()

    img = input_node(args.image, width_cm=args.width, height_cm=args.height, x=args.x)
    arch = build_arch(img)

    namefile = os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join(os.path.dirname(__file__), namefile + '.tex')
    print(f"Generating LaTeX diagram at: {output_path}")
    to_generate(arch, output_path)

if __name__ == "__main__":
    main()
