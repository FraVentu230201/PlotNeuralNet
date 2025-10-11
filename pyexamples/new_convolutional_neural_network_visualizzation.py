import os, sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pycore.tikzeng import *


def input_node(img_path, width_cm=8.0, height_cm=8.0, name='inp', x=-3.0, y=0.0, z=0.0):
    """
    Crea il nodo TikZ con l'immagine di input, gestendo percorsi con spazi.
    """
    base = os.path.dirname(__file__)
    rel = os.path.relpath(img_path, start=base).replace('\\', '/')
    return rf'\node[canvas is zy plane at x=0] ({name}) at ({x},{y},{z}) ' \
           rf'{{\includegraphics[width={width_cm}cm,height={height_cm}cm]' \
           rf'{{\detokenize{{{rel}}}}}}};'


def build_arch(img_node):
    """
    Mantiene struttura e colori dell'esempio originale, ma riduce
    il numero di bande in ciascun blocco convoluzionale.
    """
    return [
        to_head('..'),
        to_cor(),

        # Stessi colori dell'esempio originale
        r'\def\ConvColor{rgb:yellow,5;red,2.5;white,5}',
        r'\def\ConvReluColor{rgb:yellow,5;red,5;white,5}',
        r'\def\PoolColor{rgb:red,1;black,0.3}',

        r'\def\DcnvColor{rgb:blue,5;green,2.5;white,5}',
        r'\def\SoftmaxColor{rgb:magenta,5;black,7}',
        to_begin(),

        # immagine di input (stessa logica dell'originale)
        img_node,

        # conv1 (ridotto: 2 bande) + pool1
        r'\pic[shift={(0,0,0)}] at (0,0,0) {RightBandedBox={name=cr1,caption=conv1,',
        r'xlabel={{"64","64"}},zlabel=I,fill=\ConvColor,bandfill=\ConvReluColor,',
        r'height=40,width={2,2},depth=40}};',
        r'\pic[shift={(0,0,0)}] at (cr1-east) {Box={name=p1,fill=\PoolColor,opacity=0.5,height=35,width=1,depth=35}};',

        # conv2 (ridotto: 3 bande) + pool2
        r'\pic[shift={(2,0,0)}] at (p1-east) {RightBandedBox={name=cr2,caption=conv2,',
        r'xlabel={{"64","64","64"}},zlabel=I/2,fill=\ConvColor,bandfill=\ConvReluColor,',
        r'height=35,width={3,3,3},depth=35}};',
        r'\pic[shift={(0,0,0)}] at (cr2-east) {Box={name=p2,fill=\PoolColor,opacity=0.5,height=30,width=1,depth=30}};',

        # conv3 (ridotto: 4 bande) + pool3
        r'\pic[shift={(2,0,0)}] at (p2-east) {RightBandedBox={name=cr3,caption=conv3,',
        r'xlabel={{"256","256","256","256"}},zlabel=I/4,fill=\ConvColor,bandfill=\ConvReluColor,',
        r'height=30,width={4,4,4,4},depth=30}};',
        r'\pic[shift={(0,0,0)}] at (cr3-east) {Box={name=p3,fill=\PoolColor,opacity=0.5,height=23,width=1,depth=23}};',

        # conv4 (ridotto: 5 bande) + pool4
        r'\pic[shift={(1.8,0,0)}] at (p3-east) {RightBandedBox={name=cr4,caption=conv4,',
        r'xlabel={{"512","512","512","512","512"}},zlabel=I/8,fill=\ConvColor,bandfill=\ConvReluColor,',
        r'height=23,width={7,7,7,7,7},depth=23}};',
        r'\pic[shift={(0,0,0)}] at (cr4-east) {Box={name=p4,fill=\PoolColor,opacity=0.5,height=15,width=1,depth=15}};',

        # conv5 (ridotto: 6 bande) + pool5
        r'\pic[shift={(1.5,0,0)}] at (p4-east) {RightBandedBox={name=cr5,caption=conv5,',
        r'xlabel={{"512","512","512","512","512","512"}},zlabel=I/16,fill=\ConvColor,bandfill=\ConvReluColor,',
        r'height=15,width={7,7,7,7,7,7},depth=15}};',
        r'\pic[shift={(0,0,0)}] at (cr5-east) {Box={name=p5,fill=\PoolColor,opacity=0.5,height=10,width=1,depth=10}};',

        # conv6 (ridotto: 6 bande) + pool6
        r'\pic[shift={(1.2,0,0)}] at (p5-east) {RightBandedBox={name=cr6,caption=conv6,',
        r'xlabel={{"512","512","512","512","512","512"}},zlabel=I/32,fill=\ConvColor,bandfill=\ConvReluColor,',
        r'height=10,width={7,7,7,7,7,7},depth=10}};',
        r'\pic[shift={(0,0,0)}] at (cr6-east) {Box={name=p6,fill=\PoolColor,opacity=0.5,height=7,width=1,depth=7}};',

        # conv7 (ridotto: 6 bande) + pool7
        r'\pic[shift={(1.0,0,0)}] at (p6-east) {RightBandedBox={name=cr7,caption=conv7,',
        r'xlabel={{"512","512","512","512","512","512"}},zlabel=I/64,fill=\ConvColor,bandfill=\ConvReluColor,',
        r'height=7,width={7,7,7,7,7,7},depth=7}};',
        r'\pic[shift={(0,0,0)}] at (cr7-east) {Box={name=p7,fill=\PoolColor,opacity=0.5,height=5,width=1,depth=5}};',

        # fc -> conv (lasciato a 2, come struttura originale), fc8 -> conv
        r'\pic[shift={(1,0,0)}] at (p7-east) {RightBandedBox={name=cr8_9,caption=fc to conv,',
        r'xlabel={{"4096","4096"}},fill=\ConvColor,bandfill=\ConvReluColor,',
        r'height=5,width={10,10},depth=5}};',
        r'\pic[shift={(1,0,0)}] at (cr8_9-east) {Box={name=c8,caption=fc8 to conv,',
        r'xlabel={{"K","dummy"}},fill=\ConvColor,height=5,width=2,depth=5,zlabel=I/128}};',

        # deconv finale + softmax (identici come struttura e colori)
        r'\pic[shift={(2.5,0,0)}] at (c8-east) {Box={name=d32,caption=Deconv,',
        r'xlabel={{"K","dummy"}},fill=\DcnvColor,height=40,width=2,depth=40}};',
        r'\pic[shift={(1,0,0)}] at (d32-east) {Box={name=softmax,caption=softmax,',
        r'xlabel={{"K","dummy"}},fill=\SoftmaxColor,height=40,width=2,depth=40,zlabel=I}};',

        # connessioni come nell'originale
        r'\draw [connection] (p1-east) -- node {\midarrow} (cr2-west);',
        r'\draw [connection] (p2-east) -- node {\midarrow} (cr3-west);',
        r'\draw [connection] (p3-east) -- node {\midarrow} (cr4-west);',
        r'\draw [connection] (p4-east) -- node {\midarrow} (cr5-west);',
        r'\draw [connection] (p5-east) -- node {\midarrow} (cr6-west);',
        r'\draw [connection] (p6-east) -- node {\midarrow} (cr7-west);',
        r'\draw [connection] (p7-east) -- node {\midarrow} (cr8_9-west);',
        r'\draw [connection] (cr8_9-east) -- node {\midarrow} (c8-west);',
        r'\draw [connection] (c8-east) -- node {\midarrow} (d32-west);',
        r'\draw [connection] (d32-east) -- node {\midarrow} (softmax-west);',
        r'\draw[densely dashed]'
        r'(c8-nearnortheast) -- (d32-nearnorthwest)'
        r'(c8-nearsoutheast) -- (d32-nearsouthwest)'
        r'(c8-farsoutheast)  -- (d32-farsouthwest)'
        r'(c8-farnortheast)  -- (d32-farnorthwest)'
        r';',
        to_end(),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Percorso dell'immagine di ingresso")
    ap.add_argument("--width",  type=float, default=8.0, help="Larghezza immagine (cm)")
    ap.add_argument("--height", type=float, default=8.0, help="Altezza immagine (cm)")
    ap.add_argument("--x", type=float, default=-3.0, help="Posizione X dell'immagine")
    ap.add_argument("--y", type=float, default=0.0,  help="Posizione Y dell'immagine")
    ap.add_argument("--z", type=float, default=0.0,  help="Posizione Z dell'immagine")
    args = ap.parse_args()

    img = input_node(args.image, width_cm=args.width, height_cm=args.height,
                     x=args.x, y=args.y, z=args.z)
    arch = build_arch(img)

    namefile = os.path.splitext(os.path.basename(__file__))[0]
    # Usa writer locale che normalizza le backslash senza loop infinito
    def _write_tex(lines, path):
        with open(path, 'w') as f:
            for c in lines:
                # normalizza: 4 backslash -> 2, poi 2 -> 1 (usa chr(92) per evitare problemi di quoting)
                s = c.replace('\\\\', '\\').replace('\\', chr(92))
                f.write(s)
    _write_tex(arch, namefile + '.tex')
    print(f"Generato: {namefile}.tex")


if __name__ == "__main__":
    main()
