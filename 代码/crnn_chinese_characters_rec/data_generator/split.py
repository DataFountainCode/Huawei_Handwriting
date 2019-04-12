import mmcv
import numpy as np

def main():
    anno_path = './gen_anno.pickle'

    annos = mmcv.load(anno_path)

    ind = np.arange(0, len(annos))
    np.random.shuffle(ind)

    tr_end = int(0.9*len(annos))

    tr_annos = [annos[x] for x in ind[:tr_end]]
    vl_annos = [annos[x] for x in ind[tr_end:]]

    mmcv.dump(tr_annos, 'train_anno.pickle')
    mmcv.dump(vl_annos, 'valid_anno.pickle')


if __name__ == '__main__':
    main()
