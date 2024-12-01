import torch


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.if_rgb = False
        self.if_mask = False
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        if self.if_rgb:
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in parameters.items() if 'scratch.output_conv.4' not in k}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif self.if_mask:
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in parameters.items()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            self.load_state_dict(parameters)
