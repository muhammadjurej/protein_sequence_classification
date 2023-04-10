import os

class Confiq:

    def __init__(self):
        self.ROOT_PATH = "dataset/"
        self.TRAIN_PATH = os.path.join(self.ROOT_PATH, "train/")
        self.VAL_PATH = os.path.join(self.ROOT_PATH, "dev/")
        self.TEST_PATH = os.path.join(self.ROOT_PATH, "test/")
        self.batch_size = 32
        self.number_family = 250

        #Amino Acids most common use
        self.AMINO = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


    