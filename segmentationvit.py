import torch.nn as nn
from models_mae import MaskedAutoencoderViT

class SegmentationViT(MaskedAutoencoderViT):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Remove the decoder_pred layer from the original MAE model
        del self.decoder_pred

        # Create a new convolutional layer for the segmentation head
        decoder_embed_dim = self.decoder_norm.normalized_shape[0]
        self.final_conv = nn.Conv2d(decoder_embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x, _, _ = self.forward_encoder(x, mask_ratio=0)
        x = self.forward_decoder(x, ids_restore=None)  # [N, L, p*p*3]

        # Reshape the output to apply the segmentation head
        B, L, _ = x.shape
        H, W = self.patch_embed.img_size
        x = x.transpose(1, 2).view(B, -1, H, W)  # [N, C, H, W]

        # Apply the segmentation head
        x = self.final_conv(x)

        return x
