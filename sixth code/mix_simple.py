#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 去掉位置编码部分，去掉后准确率相差不大
import torch
import torch.nn as nn

def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=2, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))

        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)

        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.dep_conv1 = nn.Sequential(
            nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, self.out_planes,
                      kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1, stride=stride),
            nn.BatchNorm2d(self.out_planes),
            nn.ReLU(),
            nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(self.out_planes),
            nn.ReLU(),
            nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(self.out_planes),
            nn.ReLU()
        )
        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)
        for m in self.dep_conv1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)


        unfold_k = self.unfold(self.pad_att(k_att))
        h_out = (h + 2 * self.padding_att - self.kernel_att) // self.stride + 1
        w_out = (w + 2 * self.padding_att - self.kernel_att) // self.stride + 1
        unfold_k = unfold_k.view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)

        att = (q_att.unsqueeze(2) * unfold_k).sum(1)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv1 = self.dep_conv(f_conv)
        out_conv2 = self.dep_conv1(f_conv)

        out_conv = (out_conv1 + out_conv2) / 2

        return self.rate1 * out_att + self.rate2 * out_conv

