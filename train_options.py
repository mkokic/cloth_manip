import argparse


class TrainOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--mode', required=True, type=str,
                                 choices=['train', 'test'], default='train')
        self.parser.add_argument('--problem', required=True, type=str,
                                 choices=['posh', 'heatmap'], default='posh')
        self.parser.add_argument('--cont', type=int, default=0)
        self.parser.add_argument('--bs', type=int, default=32)
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--num', type=float, default=1024)
        self.parser.add_argument('--data_path', type=str,
                                 default='/home/code/cloth-bullet-extensions/bobak/data/')

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt, unknown = self.parser.parse_known_args()
        return opt
