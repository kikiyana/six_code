import torch
import torch.nn as nn
import torch.nn.functional as F

def position(H, W):
    loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
    loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

def stride(x, stride):
    return x[:, :, ::stride, ::stride]

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)

class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.self_attention = SelfAttention(in_planes, out_planes, kernel_att, head, stride, dilation)
        self.convolution = Convolution(in_planes, out_planes, kernel_conv, stride)
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)

    def forward(self, x):
        out_att = self.self_attention(x)
        out_conv = self.convolution(x)
        return self.rate1 * out_att + self.rate2 * out_conv

# Now we need to define the missing parts in the SelfAttention and Convolution modules.

class SelfAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att, head, stride, dilation):
        super(SelfAttention, self).__init__()
        self.head = head
        self.kernel_att = kernel_att
        self.stride = stride
        self.dilation = dilation
        self.head_dim = out_planes // head

        self.scale = (self.head_dim) ** -0.5
        self.conv_q = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_k = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_v = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        pe = self.conv_p(position(h, w))
        q = self.conv_q(x).view(b * self.head, self.head_dim, h, w) * self.scale
        k = self.conv_k(x).view(b * self.head, self.head_dim, h, w)
        v = self.conv_v(x).view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q = stride(q, self.stride)
            pe = stride(pe, self.stride)

        unfold_k = self.unfold(self.pad_att(k)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att, -1)
        unfold_k = unfold_k.permute(0, 3, 1, 2)  # b * head, H_out*
        unfold_v = self.unfold(self.pad_att(v)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att, -1)
        unfold_v = unfold_v.permute(0, 3, 1, 2)

        q = q.view(b * self.head, self.head_dim, -1).permute(0, 2, 1)  # b * head, H_out*W_out, head_dim
        pe = pe.view(self.head_dim, -1).permute(1, 0)  # H_out*W_out, head_dim

        att = torch.bmm(q, unfold_k)  # b * head, H_out*W_out, kernel_att^2
        att = att + pe.unsqueeze(1) - pe.unsqueeze(2)
        att = self.softmax(att).view(-1, self.kernel_att * self.kernel_att, h, w)

        out_att = torch.bmm(att, unfold_v.permute(0, 2, 1))  # b * head, H_out*W_out, head_dim
        out_att = out_att.permute(0, 2, 1).view(b, self.head * self.head_dim, h, w)
        return out_att

class Convolution(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_conv, stride):
        super(Convolution, self).__init__()
        self.kernel_conv = kernel_conv
        self.out_planes = out_planes
        self.fc = nn.Conv2d(3 * out_planes, kernel_conv * kernel_conv, kernel_size=1, bias=False)
        self.dep_conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_conv, padding=1, groups=in_planes, stride=stride, bias=False)
        self.dep_conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_conv, padding=1, groups=out_planes, stride=1, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        init_rate_0(self.dep_conv1.bias)
        init_rate_0(self.dep_conv2.bias)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.unsqueeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv1.weight = nn.Parameter(kernel, requires_grad=True)
        self.dep_conv2.weight = nn.Parameter(kernel, requires_grad=True)
    def forward(self, x):
        b, c, h, w = x.shape
        weights = self.fc(x).view(b, self.out_planes, self.kernel_conv * self.kernel_conv, h, w)
        weights = F.softmax(weights, dim=2)

        x = x.view(b, 1, c, h, w).repeat(1, self.kernel_conv * self.kernel_conv, 1, 1, 1)
        x = x.view(b * self.kernel_conv * self.kernel_conv, c, h, w)

        # Apply the first depthwise separable convolution
        x = self.dep_conv1(x)
        # Apply a ReLU non-linearity
        x = F.relu(x)

        # Apply the second depthwise separable convolution
        out = self.dep_conv2(x)

        # Reshape the output and weights for the weighted sum operation
        out = out.view(b, self.kernel_conv * self.kernel_conv, self.out_planes, h, w)
        out = torch.mul(out, weights).sum(dim=1)

        return out

if __name__ == '__main__':
    in_planes = 64
    out_planes = 64
    kernel_att = 7
    head = 4
    kernel_conv = 3
    stride = 1
    dilation = 1

    model = ACmix(in_planes, out_planes, kernel_att, head, kernel_conv, stride, dilation)
    input = torch.randn([2,3,224,224]).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)
    # print(summary(model, torch.zeros((1, 3, 224, 224)).cuda()))