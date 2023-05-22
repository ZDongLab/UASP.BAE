import torch
import torch.nn as nn
from inplace_abn import InPlaceABN, ABN
from einops import rearrange
from einops.layers.torch import Rearrange
from collections import OrderedDict

def InplacABN_to_ABN(module: nn.Module) -> nn.Module:
    # convert all InplaceABN layer to bit-accurate ABN layers.
    if isinstance(module, InPlaceABN):
        module_new = ABN(module.num_features, activation=module.activation,
                         activation_param=module.activation_param)
        for key in module.state_dict():
            module_new.state_dict()[key].copy_(module.state_dict()[key])
        module_new.training = module.training
        module_new.weight.data = module_new.weight.abs() + module_new.eps
        return module_new
    for name, child in reversed(module._modules.items()):
        new_child = InplacABN_to_ABN(child)
        if new_child != child:
            module._modules[name] = new_child
    return module

def conv2d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
                  bias=False),
        InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param)
    )

#conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        #nn.BatchNorm2d(oup),
        #nn.GELU()
        InPlaceABN(num_features=oup, activation="leaky_relu", activation_param=1e-2)
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                #nn.GELU(),
                InPlaceABN(num_features=hidden_dim, activation="leaky_relu", activation_param=1e-2),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                #nn.GELU(),
                InPlaceABN(num_features=hidden_dim, activation="leaky_relu", activation_param=1e-2),
                
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                #nn.GELU(),
                InPlaceABN(num_features=hidden_dim, activation="leaky_relu", activation_param=1e-2),
                
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        ##x = x + self.ff(x) ##
        x = x + self.ff(x)
        return x


class CSAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000, block_types=['C', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.stem = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.c1 = self._make_layer(block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.c2 = self._make_layer(block[block_types[0]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))

        self.c3 = self._make_layer(block[block_types[0]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.c4 = self._make_layer(block[block_types[0]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))
        self.c5 = self._make_layer(block[block_types[0]], channels[4], channels[5], num_blocks[5], (ih // 64, iw // 64))
        
        self.t3 = self._make_layer(block[block_types[1]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.t4 = self._make_layer(block[block_types[1]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))
        self.t5 = self._make_layer(block[block_types[1]], channels[4], channels[5], num_blocks[5], (ih // 64, iw // 64))
        
        self.pool = nn.AvgPool2d(ih // 64, 1)
        self.num_features = channels[-1]
        #fc = nn.Linear(channels[-1], num_classes, bias=False)
        fc = nn.Linear(channels[5], num_classes, bias=False)
        self.head = nn.Sequential(OrderedDict([('fc', fc)]))

    def forward(self, x):
        x = self.stem(x)
        x = self.c1(x)
        x = self.c2(x)
        x = (self.c3(x)).add_(self.t3(x))/2
        x = (self.c4(x)).add_(self.t4(x))/2
        x = (self.c5(x)).add_(self.t5(x))/2
        x = self.pool(x).view(-1, x.shape[1])
        x = self.head(x)
        return x

    def forward_features(self, x):
        x = self.stem(x)
        x = self.c1(x)
        x = self.c2(x)
        x = (self.c3(x)).add_(self.t3(x))/2
        x = (self.c4(x)).add_(self.t4(x))/2
        x = (self.c5(x)).add_(self.t5(x))/2

        x = self.pool(x).view(-1, x.shape[1])
        
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

def UASP_BAE_net0(model_params):
    in_chans = model_params['args'].in_chans
    num_classes = model_params['num_classes']
    image_size =   model_params['image_size']  
    num_blocks = [2, 2, 3, 5, 2, 2]            # L
    channels = [32, 64, 128, 256, 512, 1024]      # D
    return CSAtNet((image_size, image_size), in_chans, num_blocks, channels, num_classes=num_classes)


def UASP_BAE_net1(model_params):
    in_chans = model_params['args'].in_chans
    num_classes = model_params['num_classes']
    image_size =   model_params['image_size']  
    num_blocks = [2, 2, 6, 14, 2, 2]           # L
    channels = [64, 96, 192, 384, 768,1536]      # D
    return CSAtNet(image_size, in_chans, num_blocks, channels, num_classes=num_classes)

def UASP_BAE_net2(model_params):
    in_chans = model_params['args'].in_chans
    num_classes = model_params['num_classes']
    image_size =   model_params['image_size']
    num_blocks = [2, 2, 6, 14, 2, 2]           # L
    channels = [128, 128, 256, 512, 1024, 2048]   # D
    return CSAtNet((image_size, image_size), in_chans, num_blocks, channels, num_classes=num_classes)


def UASP_BAE_net3(model_params):
    in_chans = model_params['args'].in_chans
    num_classes = model_params['num_classes'] 
    image_size =   model_params['image_size'] 
    num_blocks = [2, 2, 6, 14, 2, 2]           # L
    channels = [192, 192, 384, 768, 1536, 2048]   # D
    return CSAtNet((image_size, image_size), in_chans, num_blocks, channels, num_classes=num_classes)


def UASP_BAE_net4(model_params):
    in_chans = model_params['args'].in_chans
    num_classes = model_params['num_classes']
    image_size =   model_params['image_size']    
    num_blocks = [2, 2, 12, 28, 2, 2]          # L
    channels = [192, 192, 384, 768, 1536, 2048]   # D
    return CSAtNet((image_size, image_size), in_chans, num_blocks, channels, num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


'''
class args():
    def __init__(self,in_chans):
        self.in_chans=in_chans
from torchsummary import summary
if __name__ == '__main__':
    img = torch.randn(1, 3, 448, 448)
    argss = args(3)
    model_params={'args':argss, 'num_classes':1, 'image_size':448,}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UASP_BAE_net3(model_params).to(device)
    img = img.to(device)
    #summary(net, input_size=(3, 448, 448))
    out = net(img)
    print(out.shape, count_parameters(net))
'''
