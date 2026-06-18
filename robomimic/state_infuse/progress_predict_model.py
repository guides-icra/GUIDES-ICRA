import torch
import torch.nn as nn
from torchvision import models


class ValueResNetWithAttnPerformance(nn.Module):
    def __init__(self, text_embedding_dim=768, attn_heads=2, freeze_resnet=True):
        super(ValueResNetWithAttnPerformance, self).__init__()

        # Load ResNet and freeze its layers (optional)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet_fc_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the last fully connected layer

        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False  # Freeze ResNet layers

        # Text processing network with additional layers and normalization
        text_out_dim = 256  # Reduced output dimension for text embedding
        self.text_fc = nn.Sequential(
            nn.Linear(text_embedding_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, text_out_dim),
            nn.ReLU(),
            nn.LayerNorm(text_out_dim),
            nn.Dropout(p=0.3)
        )

        # Reduce image feature size before concatenation
        self.image_projector = nn.Linear(3 * self.resnet_fc_in_features, text_out_dim)

        # Multi-head attention with reduced heads and dimensions
        self.multihead_attn = nn.MultiheadAttention(embed_dim=text_out_dim, num_heads=attn_heads)

        # Fully connected layers with residual connections
        combined_dim = text_out_dim * 2  # Both image and text now have the same dimension
        self.fc1_double = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.3)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.3)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, 1)
        )

    def forward(self, image1, image2, image3, text_embedding):
        # Combine the three images into one batch to pass through ResNet

        images = torch.cat([image1, image2, image3], dim=0)
        image_features = self.resnet(images)

        # Split back the features for each image (batch-first dimension)
        image_features = torch.split(image_features, image1.size(0), dim=0)

        # Concatenate image features
        image_features = torch.cat(image_features, dim=1)
        image_features_proj = self.image_projector(image_features)

        if text_embedding is not None:
            text_features = self.text_fc(text_embedding)

            # Multi-head attention between text and image features
            attn_output, _ = self.multihead_attn(text_features.unsqueeze(0),
                                                 image_features_proj.unsqueeze(0),
                                                 image_features_proj.unsqueeze(0))
            attn_output = attn_output.squeeze(0)

            # Concatenate image and attention output
            concatenated = torch.cat((image_features_proj, attn_output), dim=1)
            x = self.fc1_double(concatenated)
        else:
            x = self.fc1_double(image_features_proj)  # Process image features only

        x = self.fc2(x)
        x = self.fc3(x)

        return x
