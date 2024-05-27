import torch
from torch import nn
from math import sqrt, sin, cos


def scaled_dot_product_attention(Q, K, V, scale):
    return nn.functional.softmax(Q @ K.transpose(1, 2) / scale, dim=2) @ V

def scaled_dot_product_attention_multihead(Q, K, V, scale):
    return nn.functional.softmax(Q @ K.transpose(2, 3) / scale, dim=3) @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()

        self.W_Q = nn.Parameter(torch.randn((h, d_model, d_k)))
        self.W_K = nn.Parameter(torch.randn((h, d_model, d_k)))

        self.W_V = nn.Parameter(torch.randn((h, d_model, d_v)))

        self.W_O = nn.Parameter(torch.randn((h * d_v, d_model)))

        self.scale = sqrt(d_k)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):

        batch_size = Q.size(0)
        seq_len = Q.size(1)

        Q = Q.unsqueeze(1)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)

        attention_res = scaled_dot_product_attention_multihead(Q @ self.W_Q, K @ self.W_K, V @ self.W_V, self.scale)

        return attention_res.transpose(1, 2).reshape(batch_size, seq_len, -1) @ self.W_O

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, d_k, d_v, h)
        self.layer_norm_1 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.conv_before_attention = nn.Conv1d(d_model, d_model, 3, padding=1)

    def forward(self, X):

        X = self.multi_head_attention(X, X, X) + X
        X = self.layer_norm_1(X)
        X = self.feed_forward(X) + X
        return self.layer_norm_2(X)


class TransformerEncoder(nn.Module):
    def __init__(self, num_embeddings, padding_idx, d_model, d_k, d_v, h, d_ff, N, seq_len):
        super().__init__()
        self.input_embedding = nn.Embedding(num_embeddings, d_model, padding_idx=padding_idx)

        self.transformer_encoder_block_list = nn.ModuleList([
            TransformerEncoderBlock(d_model, d_k, d_v, h, d_ff) for _ in range(N)])

        
        positional_encoding = torch.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for d in range(d_model):
                if d % 2 == 0:
                    positional_encoding[pos, d] = sin(pos / (10000 ** (d / d_model)))
                else:
                    positional_encoding[pos, d] = cos(pos / (10000 ** ((d - 1) / d_model)))
        self.positional_encoding = nn.Parameter(positional_encoding)
        self.positional_encoding.requires_grad_(False)

        
    def forward(self, X):
        X = self.input_embedding(X)
        X += self.positional_encoding
        for transformer_encoder_block in self.transformer_encoder_block_list:
            X = transformer_encoder_block(X)
        return X

class ConvNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # v14
        self.conv_block_mid_1 = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)

        x = self.conv_block_mid_1(x)

        x = self.conv_block_2(x)
        return x


class SimpleResConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.batch_norm_1 = nn.BatchNorm2d(channels)
        self.conv_layer_2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.batch_norm_2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        out = self.conv_layer_1(x)
        out = self.batch_norm_1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv_layer_2(out)
        out = self.batch_norm_2(out)
        out = out + x
        out = torch.nn.functional.relu(out)
        return out


class ResConvDecoder(nn.Module):
    def __init__(self, in_channels, block_channels, block_num):
        super().__init__()

        self.conv_block_in = nn.Sequential(
            nn.Conv2d(in_channels, block_channels, 3, padding=1),
            nn.BatchNorm2d(block_channels),
            nn.ReLU()
        )

        self.res_conv_block_list = nn.ModuleList(
            [SimpleResConvBlock(block_channels, kernel_size=3, padding=1) for _ in range(block_num)])

        self.conv_block_out = nn.Sequential(
            nn.Conv2d(block_channels, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv_block_in(x)
        for block in self.res_conv_block_list:
            x = block(x)
        x = self.conv_block_out(x)
        return x

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.conv_layer_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv_layer_1(x)
        out = self.batch_norm_1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv_layer_2(out)
        out = self.batch_norm_2(out)
        out = torch.nn.functional.relu(out)
        return out


class UNetUpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample_layer = nn.Upsample(scale_factor=2)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.up_sample_layer(x)
        out = self.conv_layer(out)
        out = self.batch_norm(out)
        out = torch.nn.functional.relu(out)
        return out


class UNetEncoder(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        self.contracting_path = nn.ModuleList([UNetConvBlock(channels_list[0], channels_list[1])])
        for i in range(1, len(channels_list) - 1):
            self.contracting_path.append(nn.Sequential(
                nn.MaxPool2d(2),
                UNetConvBlock(channels_list[i], channels_list[i + 1])
            ))
        
    def forward(self, x):
        skipped_connections = []
        for conv_block in self.contracting_path:
            x = conv_block(x)
            skipped_connections.append(x)
        return x, skipped_connections[:-1]
    

class UNetDecoder(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        self.expansive_path = nn.ModuleList([])
        for i in range(len(channels_list) - 1):
            self.expansive_path.append(nn.ModuleList([
                UNetUpConv(channels_list[i], channels_list[i + 1]),
                UNetConvBlock(channels_list[i + 1] * 2, channels_list[i + 1])
            ]))

    def forward(self, x, skipped_connections):
        skipped_connections.reverse()
        for i in range(len(self.expansive_path)):
            up_sample, conv = self.expansive_path[i]
            x = up_sample(x)
            
            if x.shape[-1] < skipped_connections[i].shape[-1]:
                padded_x = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + 1, x.shape[3] + 1)).float().to(x.device)
                padded_x[:, :, :-1, :-1] = x
                x = padded_x
            x = conv(torch.cat((x, skipped_connections[i]), dim=1))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        decoder_out_channels = 64

        encoder_channels_list = [in_channels, decoder_out_channels, 128, 256, 512, 1024]
        decoder_channels_list = [1024, 512, 256, 128, decoder_out_channels]

        self.encoder = UNetEncoder(encoder_channels_list)
        self.decoder = UNetDecoder(decoder_channels_list)

        self.out = nn.Conv2d(decoder_out_channels, out_channels, 1, padding=0)

    def forward(self, x):
        latent, skipped_connections = self.encoder(x)
        latent = self.decoder(latent, skipped_connections)
        y = self.out(latent)
        return y

class Model(nn.Module):
    def __init__(self, num_embeddings, padding_idx, d_model, d_k, d_v, h, d_ff, N, seq_len, disable_rcsa=False,
                 use_low_high_resolution=False, low_high_resolution_patch_size=1):
        super().__init__()

        self.transformer_encoder = TransformerEncoder(num_embeddings, padding_idx, d_model, d_k, d_v, h, d_ff, N, seq_len)

        self.conv_net = UNet(2 * d_model, 1)

        mask_no_sharp_loops = torch.ones((seq_len, seq_len), dtype=torch.float)
        for i in range(seq_len):
            l_idx = i - 3
            if l_idx < 0:
                l_idx = 0
            r_idx = i + 3

            if r_idx > seq_len - 1:
                r_idx = seq_len - 1

            r_idx += 1
            mask_no_sharp_loops[i, l_idx:r_idx] = 0.0
        self.mask_no_sharp_loops = nn.Parameter(mask_no_sharp_loops)
        self.mask_no_sharp_loops.requires_grad_(False)

        # discard
        self.disable_rcsa = disable_rcsa
        # discard
        self.use_low_high_resolution = use_low_high_resolution
        # discard
        self.low_high_resolution_patch_size = low_high_resolution_patch_size
        # discard
        self.eval_confident_threshold = -1.0

    def forward(self, x: torch.Tensor):
        x = self.transformer_encoder(x)
        batch_size, seq_len, d_model = x.size()

        x = x.unsqueeze(1)

        x = x.repeat((1, seq_len, 1, 1))

        x = torch.cat([x, x.transpose(1, 2)], dim=3)

        x = x.permute((0, 3, 1, 2))

        if self.use_low_high_resolution:
            x = torch.nn.functional.avg_pool2d(x, self.low_high_resolution_patch_size)

        x = self.conv_net(x)

        x = x.squeeze(1)

        x = (x + x.transpose(1, 2)) / 2
   
        if self.training:
            return x
        else:
            ca = torch.zeros_like(x)
            ra = torch.zeros_like(x)

            row_idx_for_c = torch.argmax(x, dim=1)
            column_idx_for_r = torch.argmax(x, dim=2)

            ca[torch.arange(batch_size).reshape((-1, 1)), row_idx_for_c, torch.arange(x.size(1))] = 1.0
            ra[torch.arange(batch_size).reshape((-1, 1)), torch.arange(x.size(1)), column_idx_for_r] = 1.0

            x_softmax = torch.nn.functional.softmax(x, dim=2)
            x_softmax_larger_than_threshold = (x_softmax > self.eval_confident_threshold).float()
            confident_selection = x_softmax_larger_than_threshold * x_softmax_larger_than_threshold.transpose(1, 2)

            x = ra * ca

            x = x * confident_selection

            if not self.use_low_high_resolution:
                x = x * self.mask_no_sharp_loops
            return x


if __name__ == '__main__':
    pass
