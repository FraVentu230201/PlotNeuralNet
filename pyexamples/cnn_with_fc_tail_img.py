# pyexamples/cnn_with_fc_tail_img.py
import os, sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pycore.tikzeng import *


def input_node(img_path, width_cm=8.0, height_cm=8.0, name="img", x=-3.0, y=0.0, z=0.0):
    """Crea il nodo TikZ con l'immagine di input, gestendo percorsi con spazi."""
    base = os.path.dirname(__file__)
    rel = os.path.relpath(img_path, start=base).replace('\\', '/')
    return (
        rf'\node[canvas is zy plane at x=0] ({name}) at ({x},{y},{z}) '
        rf'{{\includegraphics[width={width_cm}cm,height={height_cm}cm]'
        rf'{{\detokenize{{{rel}}}}}}};'
    )


def build_arch(img_node, fc_dims, add_flatten):
    arch = [
        to_head('..'),
        to_cor(),
        r'\def\DcnvColor{rgb:blue,5;green,2.5;white,5}',
        to_begin(),
        img_node,
        # Backbone conv/pool fino a p5
        r'\pic[shift={(0,0,0)}] at (0,0,0) {RightBandedBox={name=cr1,caption=conv1,'
        r'xlabel={{"64","64"}},zlabel=I,fill=\ConvColor,bandfill=\ConvReluColor,'
        r'height=40,width={2,2},depth=40}};',
        r'\pic[shift={(0,0,0)}] at (cr1-east) {Box={name=p1,fill=\PoolColor,opacity=0.5,height=35,width=1,depth=35}};',
        r'\pic[shift={(2,0,0)}] at (p1-east) {RightBandedBox={name=cr2,caption=conv2,'
        r'xlabel={{"64","64"}},zlabel=I/2,fill=\ConvColor,bandfill=\ConvReluColor,'
        r'height=35,width={3,3},depth=35}};',
        r'\pic[shift={(0,0,0)}] at (cr2-east) {Box={name=p2,fill=\PoolColor,opacity=0.5,height=30,width=1,depth=30}};',
        r'\pic[shift={(2,0,0)}] at (p2-east) {RightBandedBox={name=cr3,caption=conv3,'
        r'xlabel={{"256","256","256"}},zlabel=I/4,fill=\ConvColor,bandfill=\ConvReluColor,'
        r'height=30,width={4,4,4},depth=30}};',
        r'\pic[shift={(0,0,0)}] at (cr3-east) {Box={name=p3,fill=\PoolColor,opacity=0.5,height=23,width=1,depth=23}};',
        r'\pic[shift={(1.8,0,0)}] at (p3-east) {RightBandedBox={name=cr4,caption=conv4,'
        r'xlabel={{"512","512","512"}},zlabel=I/8,fill=\ConvColor,bandfill=\ConvReluColor,'
        r'height=23,width={7,7,7},depth=23}};',
        r'\pic[shift={(0,0,0)}] at (cr4-east) {Box={name=p4,fill=\PoolColor,opacity=0.5,height=15,width=1,depth=15}};',
        r'\pic[shift={(1.5,0,0)}] at (p4-east) {RightBandedBox={name=cr5,caption=conv5,'
        r'xlabel={{"512","512","512"}},zlabel=I/16,fill=\ConvColor,bandfill=\ConvReluColor,'
        r'height=15,width={7,7,7},depth=15}};',
        r'\pic[shift={(0,0,0)}] at (cr5-east) {Box={name=p5,fill=\PoolColor,opacity=0.5,height=10,width=1,depth=10}};',
    ]

    conn = [
        r'\draw [connection] (p1-east) -- node {\midarrow} (cr2-west);',
        r'\draw [connection] (p2-east) -- node {\midarrow} (cr3-west);',
        r'\draw [connection] (p3-east) -- node {\midarrow} (cr4-west);',
        r'\draw [connection] (p4-east) -- node {\midarrow} (cr5-west);',
    ]

    prev_anchor = 'p5'
    shift = 1.5
    fc_nodes = []
    fc_index = 1
    prev_neurons = []
    neuron_count = 5
    y_spacing = 4

    if add_flatten:
        fc_nodes.append(
            rf'\pic[shift={{({shift},0,0)}}] at ({prev_anchor}-east) '
            r'{Box={name=flatten,caption=flatten,fill=\FcColor,height=10,width=2,depth=10}};'
        )
        conn.append(rf'\draw [connection] ({prev_anchor}-east) -- node {{\midarrow}} (flatten-west);')
        prev_anchor = 'flatten'

    for dim in fc_dims:
        layer_name = 'softmax' if dim == 'K' else f'fc{fc_index}'
        color = '\\SoftmaxColor' if dim == 'K' else '\\FcColor'
        caption = 'softmax' if dim == 'K' else layer_name
        label = 'K' if dim == 'K' else dim

        new_neurons = []
        for n in range(neuron_count):
            y = (n - (neuron_count - 1) / 2) * y_spacing
            node_name = f'{layer_name}-{n+1}'
            cap = caption if n == neuron_count // 2 else ''
            fc_nodes.append(
                rf'\pic[shift={{({shift},{y},0)}}] at ({prev_anchor}-east) '
                rf'{{Ball={{name={node_name},caption={cap},fill={color},opacity=0.6,radius=2.5}}}};'
            )
            if prev_neurons:
                for p in prev_neurons:
                    conn.append(
                        rf'\draw [connection] ({p}-east) -- node {{\midarrow}} ({node_name}-west);'
                    )
            else:
                conn.append(
                    rf'\draw [connection] ({prev_anchor}-east) -- node {{\midarrow}} ({node_name}-west);'
                )
            new_neurons.append(node_name)

        fc_nodes.append(
            rf'\node[anchor=west] at ({layer_name}-{neuron_count//2 + 1}-east) {{{label}}};'
        )

        prev_neurons = new_neurons
        prev_anchor = new_neurons[len(new_neurons)//2]
        if dim != 'K':
            fc_index += 1

    arch.extend(fc_nodes)
    arch.extend(conn)
    arch.append(to_end())
    return arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path dell'immagine")
    ap.add_argument("--width", type=float, default=8.0, help="Larghezza immagine (cm)")
    ap.add_argument("--height", type=float, default=8.0, help="Altezza immagine (cm)")
    ap.add_argument("--x", type=float, default=-3.0, help="Posizione X dell'immagine")
    ap.add_argument("--fc-dims", default="4096,4096,K", help="Dimensioni dei layer FC separate da virgole; l'ultimo può essere K")
    ap.add_argument("--add-flatten", action="store_true", help="Inserisce un layer flatten prima dei FC")
    args = ap.parse_args()

    img = input_node(args.image, width_cm=args.width, height_cm=args.height, x=args.x)
    fc_dims = [d.strip() for d in args.fc_dims.split(',') if d.strip()]
    arch = build_arch(img, fc_dims, args.add_flatten)

    namefile = os.path.splitext(os.path.basename(__file__))[0]
    to_generate(arch, namefile + '.tex')


if __name__ == "__main__":
    main()

