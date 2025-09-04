import os
import sys
import argparse

# add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pycore.tikzeng import *


def input_node(img_path, width_cm=8.0, height_cm=8.0, name="img", x=-3.0, y=0.0, z=0.0):
    """Return LaTeX code for including an image node, handling spaces."""
    base = os.path.dirname(__file__)
    rel = os.path.relpath(img_path, start=base).replace('\\', '/')
    return (
        rf'\node[canvas is zy plane at x=0] ({name}) at ({x},{y},{z}) '
        rf'{{\includegraphics[width={width_cm}cm,height={height_cm}cm]{{\detokenize{{{rel}}}}}}};'
    )


def build_arch(img_node, fc_dims, add_flatten):
    if len(fc_dims) < 2:
        raise ValueError("Need at least two fully connected layer dimensions")

    arch = [
        to_head('..'),
        to_cor(),
        to_begin(),
        img_node,
        # Backbone up to p5
        r'\pic[shift={(0,0,0)}] at (0,0,0) {RightBandedBox={name=cr1,caption=conv1,xlabel={{"64","64"}},zlabel=I,fill=\ConvColor,bandfill=\ConvReluColor,height=40,width={2,2},depth=40}};',
        r'\pic[shift={(0,0,0)}] at (cr1-east) {Box={name=p1,fill=\PoolColor,opacity=0.5,height=35,width=1,depth=35}};',
        r'\pic[shift={(2,0,0)}] at (p1-east) {RightBandedBox={name=cr2,caption=conv2,xlabel={{"64","64"}},zlabel=I/2,fill=\ConvColor,bandfill=\ConvReluColor,height=35,width={3,3},depth=35}};',
        r'\pic[shift={(0,0,0)}] at (cr2-east) {Box={name=p2,fill=\PoolColor,opacity=0.5,height=30,width=1,depth=30}};',
        r'\pic[shift={(2,0,0)}] at (p2-east) {RightBandedBox={name=cr3,caption=conv3,xlabel={{"256","256","256"}},zlabel=I/4,fill=\ConvColor,bandfill=\ConvReluColor,height=30,width={4,4,4},depth=30}};',
        r'\pic[shift={(0,0,0)}] at (cr3-east) {Box={name=p3,fill=\PoolColor,opacity=0.5,height=23,width=1,depth=23}};',
        r'\pic[shift={(1.8,0,0)}] at (p3-east) {RightBandedBox={name=cr4,caption=conv4,xlabel={{"512","512","512"}},zlabel=I/8,fill=\ConvColor,bandfill=\ConvReluColor,height=23,width={7,7,7},depth=23}};',
        r'\pic[shift={(0,0,0)}] at (cr4-east) {Box={name=p4,fill=\PoolColor,opacity=0.5,height=15,width=1,depth=15}};',
        r'\pic[shift={(1.5,0,0)}] at (p4-east) {RightBandedBox={name=cr5,caption=conv5,xlabel={{"512","512","512"}},zlabel=I/16,fill=\ConvColor,bandfill=\ConvReluColor,height=15,width={7,7,7},depth=15}};',
        r'\pic[shift={(0,0,0)}] at (cr5-east) {Box={name=p5,fill=\PoolColor,opacity=0.5,height=10,width=1,depth=10}};',
    ]

    conn = [
        r'\draw [connection] (p1-east) -- node {\midarrow} (cr2-west);',
        r'\draw [connection] (p2-east) -- node {\midarrow} (cr3-west);',
        r'\draw [connection] (p3-east) -- node {\midarrow} (cr4-west);',
        r'\draw [connection] (p4-east) -- node {\midarrow} (cr5-west);',
    ]

    prev = 'p5'

    if add_flatten:
        arch.append(r'\pic[shift={(1.5,0,0)}] at (p5-east) {Box={name=flatten,caption=flatten,fill=\FcColor,height=10,width=2,depth=10}};')
        conn.append(r'\draw [connection] (p5-east) -- node {\midarrow} (flatten-west);')
        prev = 'flatten'

    fc_index = 1
    # Any intermediate FC boxes before the last two neuron columns
    for dim in fc_dims[:-2]:
        fc_name = f'fc{fc_index}'
        arch.append(rf'\pic[shift={{(1.5,0,0)}}] at ({prev}-east) {{Box={{name={fc_name},caption={fc_name},fill=\FcColor,height=10,width=2,depth=10,zlabel={dim}}}}};')
        conn.append(rf'\draw [connection] ({prev}-east) -- node {{\midarrow}} ({fc_name}-west);')
        prev = fc_name
        fc_index += 1

    # First neuron column (second to last FC layer)
    fc1_name = f'fc{fc_index}'
    y_positions = [-8.0, -4.0, 0.0, 4.0, 8.0]
    for idx, y in enumerate(y_positions, 1):
        arch.append(rf'\pic[shift={{(1.5,{y},0)}}] at ({prev}-east) {{Ball={{name={fc1_name}-{idx},caption=,fill=\FcColor,opacity=0.6,radius=2.5}}}};')
        conn.append(rf'\draw [connection] ({prev}-east) -- node {{\midarrow}} ({fc1_name}-{idx}-west);')
    arch.append(rf'\node[anchor=west] at ({fc1_name}-3-east) {{{fc_dims[-2]}}};')
    prev_nodes = [f'{fc1_name}-{i}' for i in range(1,6)]
    prev_anchor = f'{fc1_name}-3'
    fc_index += 1

    # Second neuron column (last FC layer)
    fc2_name = f'fc{fc_index}'
    for idx, y in enumerate(y_positions, 1):
        arch.append(rf'\pic[shift={{(1.5,{y},0)}}] at ({prev_anchor}-east) {{Ball={{name={fc2_name}-{idx},caption=,fill=\FcColor,opacity=0.6,radius=2.5}}}};')
        for j in range(1,6):
            conn.append(rf'\draw [connection] ({fc1_name}-{j}-east) -- node {{\midarrow}} ({fc2_name}-{idx}-west);')
    arch.append(rf'\node[anchor=west] at ({fc2_name}-3-east) {{{fc_dims[-1]}}};')

    arch.extend(conn)
    arch.append(to_end())
    return arch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path dell\'immagine')
    parser.add_argument('--width', type=float, default=8.0, help='Larghezza immagine (cm)')
    parser.add_argument('--height', type=float, default=8.0, help='Altezza immagine (cm)')
    parser.add_argument('--x', type=float, default=-3.0, help='Posizione X dell\'immagine')
    parser.add_argument('--fc-dims', default='2048,512', help='Due o pi\u00f9 dimensioni FC separate da virgola; le ultime due sono visualizzate come neuroni')
    parser.add_argument('--add-flatten', action='store_true', help='Inserisce un layer flatten prima dei FC')
    args = parser.parse_args()

    img = input_node(args.image, width_cm=args.width, height_cm=args.height, x=args.x)
    fc_dims = [d.strip() for d in args.fc_dims.split(',') if d.strip()]
    arch = build_arch(img, fc_dims, args.add_flatten)

    namefile = os.path.splitext(os.path.basename(__file__))[0]
    out_path = os.path.join(os.path.dirname(__file__), namefile + '.tex')
    to_generate(arch, out_path)

if __name__ == '__main__':
    main()
