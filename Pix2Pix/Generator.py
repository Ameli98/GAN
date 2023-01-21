import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channel=3, image_channel=3, initial_feature=64):
        super().__init__()
        # Encoder layers
        self.encoder_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, initial_feature,
                          kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
                nn.LeakyReLU(0.2),
            )
        ])
        for i in range(1, 7):
            if i <= 3:
                self.encoder_list.append(
                    encoder_block(initial_feature, initial_feature*2)
                )
                initial_feature *= 2
            else:
                self.encoder_list.append(
                    encoder_block(initial_feature, initial_feature)
                )
        self.encoder_list.append(nn.Sequential(
            nn.Conv2d(initial_feature, initial_feature, kernel_size=4,
                      stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        ))
        # Decoder layers
        self.decoder_list = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(initial_feature, initial_feature, kernel_size=4,
                                   stride=2, padding=1),
                nn.BatchNorm2d(initial_feature),
                nn.Dropout(),
                nn.ReLU(),
            )
        ])
        for j in range(1, 4):
            self.decoder_list.append(
                decoder_block(initial_feature*2, initial_feature)
            )
        initial_feature = int(initial_feature/2)
        for j in range(4, 7):
            self.decoder_list.append(
                decoder_block(initial_feature*4,
                              initial_feature, dropout=False)
            )
            initial_feature = int(initial_feature/2)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(initial_feature*2, image_channel, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # restore middle product
        encoder_output = [self.encoder_list[0](x)]
        for encoder_layer in self.encoder_list[1:]:
            encoder_output.append(
                encoder_layer(
                    encoder_output[-1]
                ))
        # skip connection
        y = encoder_output.pop()
        x = self.decoder_list[0](y)
        for decoder_layer in self.decoder_list[1:]:
            y = encoder_output.pop()
            x = torch.cat((x, y), dim=1)
            x = decoder_layer(x)
        return self.final_layer(x)


class encoder_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4,
                      stride=2, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.layer(x)


class decoder_block(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=True):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),)

        if dropout:
            self.relu_layer = nn.Sequential(
                nn.Dropout(),
                nn.ReLU(),
            )
        else:
            self.relu_layer = nn.ReLU()

    def forward(self, x):
        x = self.initial_layer(x)
        return self.relu_layer(x)


if __name__ == "__main__":
    gen = Generator(3).to("cuda")
    test_noise = torch.randn(1, 3, 256, 256).to("cuda")
    gen_img = gen(test_noise)
    print(gen_img.shape)
