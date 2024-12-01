
from src.models.depthfm.dfm import DepthFM
from src.models.depthfm.dfm_amodal import DepthFMAmodal
from src.models.amodalsynthdrive.deeplab import ADDeepLab
# from src.models.amodalsynthdrive.deeplab_dav2 import ADDeepLabDAv2
# from src.models.amodalsynthdrive.deeplab_naive_dav2 import ADDeepLabNaiveDAv2
# from src.models.amodalsynthdrive.deeplab_naive import ADDeepLabNaive
from src.models.amodalsynthdrive.dav2 import AmodalDAv2
from src.models.amodalsynthdrive.jo_amodal import PartialCompletionContentDPT
from src.models.amodalsynthdrive.invisible_stitch import InvisibleStitch


model_name_class_dict = {
    "DepthFM": DepthFM,
    "DepthFMAmodal": DepthFMAmodal,
    "ADDeepLab": ADDeepLab,
    # "ADDeepLabDAv2": ADDeepLabDAv2,
    # "ADDeepLabNaiveDAv2": ADDeepLabNaiveDAv2,
    # "ADDeepLabNaive": ADDeepLabNaive,
    "AmodalDAv2": AmodalDAv2,
    "PartialCompletionContentDPT": PartialCompletionContentDPT,
    "InvisibleStitch": InvisibleStitch}

def get_model(model_name, **kwargs):
    if model_name in model_name_class_dict.keys():
        model_class = model_name_class_dict[model_name]
        model = model_class(**kwargs)
    else:
        raise NotImplementedError

    return model
