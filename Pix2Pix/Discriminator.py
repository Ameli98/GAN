import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_channel=3, feature_list=[64, 128, 256, 512]):
        super().__init__()

        first_layer = nn.Sequential(
            nn.Conv2d(image_channel*2, feature_list[0],
                      kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),)

        middle_layers = []
        for index, in_channel in enumerate(feature_list[:-1]):
            middle_layers.append(disc_block(in_channel, feature_list[index+1]))

        last_layer = nn.Sequential(
            nn.Conv2d(feature_list[-1], feature_list[-1],
                      kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
            nn.Conv2d(feature_list[-1], 1,
                      kernel_size=4, stride=1, padding=1, padding_mode="reflect"),)

        self.layer = nn.Sequential(*first_layer, *middle_layers, *last_layer)

    def forward(self, x, y):
        # pre-processing
        x = torch.cat((x, y), dim=1)
        return self.layer(x)


class disc_block(nn.Module):
    def __init__(self, input_channel, output_channel, stride=2):
        super().__init__()
        self.default_layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel,
                      kernel_size=4, stride=stride, padding=0, padding_mode="reflect"),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.default_layer(x)


# if __name__ == "__main__":
#     disc = Discriminator()
#     test_gen = torch.randn(1, 3, 286, 286)
#     test_real = torch.randn(1, 3, 286, 286)
#     test_score = disc(test_gen, test_real)
#     print(test_score.shape)
