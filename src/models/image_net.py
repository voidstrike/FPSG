import torch
import torch.nn as nn

from torchvision.models import vgg16_bn

class ImageEncoderWarpper(nn.Module):
    def __init__(self, core='vgg_16', finetune_layer=0):
        super(ImageEncoderWarpper, self).__init__()

        self.img_feature_extractor = None
        self.img_feature_pool = None
        self.finetune_layer = finetune_layer
        if core == 'vgg_16':
            self.img_feature_extractor = vgg16_bn(pretrained=True).features
            self.img_feature_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        else:
            raise NotImplementedError(f'Unsupported Image Encoder Core Compoenent: {core}')

        pass

    def forward(self, x):
        # X should be the image, B * C * H * W
        latent_vec = self.img_feature_pool(self.img_feature_extractor(x)).squeeze(-1).squeeze(-1)
        return latent_vec  # Return a tensor with shape B * 512

    def _set_finetune(self, new_layer=None):
        if new_layer != None:
            self.finetune_layer = new_layer

        tmp_flag = self.finetune_layer
        for idx in range(len(self.img_feature_extractor)-1, -1, -1):
            if isinstance(self.img_feature_extractor[idx], nn.Conv2d):
                if tmp_flag > 0:
                    self.img_feature_extractor[idx].requires_grad_(True)
                    tmp_flag -= 1
                else:
                    self.img_feature_extractor[idx].requires_grad_(False)

        pass
