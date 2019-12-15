import argparse
import logging

from model import Model

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser('Fasttext style text classification')
    parser.add_argument('dataset', help='dataset', type=lambda t: t.split(':'))
    parser.add_argument('-prefix', help='prefix for dataset', default='', type=str)
    parser.add_argument('-tag', help='tag for experiment', default='', type=str)
    parser.add_argument('-embedding', help='word embedding', default='', type=str)
    parser.add_argument('-length', help='max sequence length', default=64, type=int)
    parser.add_argument('-count', help='max pca stage', default=0, type=int)
    parser.add_argument('-dimension', help='embedding dimension', default=100, type=int)
    parser.add_argument('-line', help='max lines', default=20000, type=int)
    parser.add_argument('-epoch_mean', help='max epoch mean', default=40, type=int)
    parser.add_argument('-lr_mean', help='learning rate for mean', default=0.5, type=float)

    args = parser.parse_args()
    log = logging.getLogger(__name__)
    log.info('Args: %s', args)

    model = Model(args)
    model.build_vocabulary()

    model.init_hdf5file(args.prefix + args.dataset[0])
    model.init_hdf5file(args.prefix + args.dataset[1])

    model.sgd()
    model.close()


if __name__ == "__main__":
    main()
