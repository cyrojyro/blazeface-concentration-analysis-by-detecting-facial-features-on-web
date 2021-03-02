import argparse
import numpy as np
import os


parser = argparse.ArgumentParser(
    description="Generate anchors")
parser.add_argument('--smin', type=float, default=0.2,
                    help='Minimum scale')
parser.add_argument('--smax', type=float, default=0.9,
                    help='Maximum scalet')
parser.add_argument('--anchor_num', nargs='+', type=int, default=[2, 6],
                    help='Anchor number of each cells')
parser.add_argument('--cell_size', nargs='+', type=int, default=[16, 8],
                    help='Cell size')
parser.add_argument('--output_path', type=str, default='./',
                    help='output directory')


def calc_anchors(s_min, s_max, anchor_num, cell_size):
    """
    [num_box, 4(cx, cy, w, h)] anchor(prior) creation.\n
    w, h ratio fixed to 1 (following Blazeface paper)
    """
    total_anchor = sum(anchor_num)

    anchors = np.empty([0, 4], dtype=np.float32)
    cell_cumulated = 0
    for iteration, anchor in enumerate(anchor_num):
        cells = cell_size[iteration]
        for y in range(cells):
            for x in range(cells):
                for order in range(anchor):
                    scale = s_min + (s_max - s_min) / \
                        (total_anchor - 1) * (order + cell_cumulated)
                    anchors = np.vstack([anchors, np.array(
                        [(x + 0.5) / cells, (y + 0.5) / cells, scale, scale])])
        cell_cumulated += anchor
    return anchors


if __name__ == "__main__":
    args = parser.parse_args()
    s_min = args.smin
    s_max = args.smax
    anchor_num = args.anchor_num
    cell_size = args.cell_size

    assert s_min < s_max
    assert len(anchor_num) == len(cell_size)

    anchors = calc_anchors(s_min, s_max, anchor_num, cell_size)

    print("last 10 results:", anchors[-10:], sep='\n')
    print("shape:", anchors.shape, sep=' ')
    np.save(os.path.join(args.output_path, "anchors.npy"), anchors)
