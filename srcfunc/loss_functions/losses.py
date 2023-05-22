from mimetypes import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

class ACLossGs(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(ACLossGs, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y, subclass=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        #x_sigmoid = torch.sigmoid(x)
        xs_pos = x
        xs_neg = 1 - x

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        y_p = y.clone()
        y_p[y_p<0.9]=0
        y_p[y_p>=0.9]=1
        los_pos = y_p * torch.log(xs_pos.clamp(min=self.eps))

        y_n = y.clone()
        y_n[y_n>0.2]=1
        y_n[y_n<=0.2]=0
        los_neg = (1 - y_n) * torch.log(xs_neg.clamp(min=self.eps))

        y_c = y.clone()
        y_c[y_c>0.9]=0
        y_c[y_c<0.2]=0
        y_ct = y_c.clone()
        y_ct[y_c>0]=1

        los_c = torch.var(y_c - xs_pos.clamp(min=self.eps))
        loss = los_pos + los_neg + los_c

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y_p
            pt1 = xs_neg * y_n  # pt = p if t > 0 else 1-p
            pt2 = y_ct
            pt = pt0 + pt1 + pt2
            one_sided_gamma = self.gamma_pos * (y_p + y_ct) + self.gamma_neg * (1 - y_n)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()