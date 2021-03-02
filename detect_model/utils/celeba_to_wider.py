import os

GT_DIR = '/home/cyrojyro/hddrive/wider_face_split/list_bbox_celeba.txt'
OUTPUT_PATH = './celeba_gt.txt'


def main():
    output_file = open(OUTPUT_PATH, 'w')
    f = open(GT_DIR, 'r')

    f.readline()
    f.readline()

    while True:
        line = f.readline().strip("\n ").split(' ')
        line = list(filter(('').__ne__, line))

        # break if end of line
        if not line:
            break

        dir = 'celeba/' + line[0] + '\n'
        gt = line[1] + ' ' + line[2] + ' ' + \
            line[3] + ' ' + line[4] + ' 0 0 0 0 0 0\n'
        output_file.write(dir)
        output_file.write('1\n')
        output_file.write(gt)

    output_file.close()


if __name__ == '__main__':
    main()
