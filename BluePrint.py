from outer.File import File
from outer.BluePrint import BluePrint


class BluePrint(BluePrint):
    def __init__(self, key='1'):
        # define file or dir
        self.LOG_MAIN = File('log_{}.log', is_dir=False)
        self.MODEL_CHECKPOINT = File('checkpoint', is_dir=True)

        self.TRAIN_GT_OUTPUT = File('train/gt', is_dir=True)
        self.TRAIN_SEG_OUTPUT = File('train/seg', is_dir=True)
        self.TRAIN_REF_OUTPUT = File('train/ref', is_dir=True)

        self.TEST_GT_OUTPUT = File('test/gt', is_dir=True)
        self.TEST_SEG_OUTPUT = File('test/seg', is_dir=True)
        self.TEST_REF_OUTPUT = File('test/ref', is_dir=True)
        self.TEST_WEIGHT_OUTPUT = File('test/weight', is_dir=True)

        self.VAL_GT_OUTPUT = File('val/gt', is_dir=True)
        self.VAL_SEG_OUTPUT = File('val/seg', is_dir=True)
        self.VAL_REF_OUTPUT = File('val/ref', is_dir=True)
        # recall father init
        super().__init__(key=key)