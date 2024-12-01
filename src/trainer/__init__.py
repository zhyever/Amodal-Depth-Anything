# Author: Bingxin Ke
# Last modified: 2024-05-17

from .depthfm_trainer import DepthFMTrainer
from .depthfm_amodal_trainer import DepthFMAmodalTrainer
from .amodalsynthdrive_trainer import AmodalSynthDriveTrainer
# from .amodalsynthdrive_naive_trainer import AmodalSynthDriveNaiveTrainer
from .discriminative_trainer import DiscriminativeTrainer
from .invisible_stitch_trainer import InvisibleStitchTrainer

trainer_cls_name_dict = {
    "DepthFMTrainer": DepthFMTrainer,
    "DepthFMAmodalTrainer": DepthFMAmodalTrainer,
    "AmodalSynthDriveTrainer": AmodalSynthDriveTrainer,
    # "AmodalSynthDriveNaiveTrainer": AmodalSynthDriveNaiveTrainer,
    "DiscriminativeTrainer": DiscriminativeTrainer,
    "InvisibleStitchTrainer": InvisibleStitchTrainer
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
