import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']


_C.DATA = CN()
_C.DATA.ROOT = ''
_C.DATA.TRAIN_FILE = ''
_C.DATA.VAL_FILE = ''
_C.DATA.DATASET = 'kinetics400'
_C.DATA.INPUT_SIZE = 224
_C.DATA.NUM_FRAMES = 8
_C.DATA.NUM_CLASSES = 400
_C.DATA.LABEL_LIST = 'labels/kinetics_400_labels.csv'


_C.MODEL = CN()
_C.MODEL.ARCH = 'ViT-B/32'
_C.MODEL.DROP_PATH_RATE = 0.
_C.MODEL.PRETRAINED = None
_C.MODEL.RESUME = None
_C.MODEL.FIX_TEXT = True


_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.003
_C.TRAIN.LR = 8.e-6
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.LR_SCHEDULER = 'cosine'
_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.OPT_LEVEL = 'O1'
_C.TRAIN.AUTO_RESUME = False
_C.TRAIN.USE_CHECKPOINT = False


_C.AUG = CN()
_C.AUG.LABEL_SMOOTH = 0.1
_C.AUG.COLOR_JITTER = 0.8
_C.AUG.GRAY_SCALE = 0.2
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5


_C.TEST = CN()
_C.TEST.NUM_CLIP = 1
_C.TEST.NUM_CROP = 1
_C.TEST.ONLY_TEST = False


_C.OUTPUT = ''
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 50
_C.SEED = 1024



def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.config)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    # merge from specific arguments
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.output:
        config.OUTPUT = args.output
    if args.only_test:
        config.TEST.ONLY_TEST = True
    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config