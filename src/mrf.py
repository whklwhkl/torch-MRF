import torch
import torch.nn as nn
import torch.nn.functional as F


class MRF(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_components):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_components = num_components
        self.in_channels = in_channels
        self.in_split = list(num_components * in_channels ** i for i in range(3))
        self.fc_gmm = nn.Conv2d(hidden_channels, sum(self.in_split), 1)
        self.fc_xh = nn.Conv2d(in_channels, hidden_channels, 1)
        self.fc_hh = nn.Conv2d(hidden_channels, hidden_channels, 1)

    def forward(self, x, h=None):
        _, _, H, W = x.shape
        xh = self.fc_xh(x)
        if h is None:
            h = torch.zeros_like(xh)
        h = self.fc_hh(h.detach())
        hl = F.pad(h, [1, 0, 0, 0])[..., :-1]
        hr = F.pad(h, [0, 1, 0, 0])[..., 1:]
        ht = F.pad(h, [0, 0, 1, 0])[:, :, :-1]
        h = torch.zeros_like(h)
        h[..., 0::2] = hl[..., 0::2]
        h[..., 1::2] = hr[..., 1::2]
        h = h + ht  # hidden states in the forward graph

        nl = F.pad(xh, [1, 0, 0, 0])[:, :, :, :-1]
        nr = F.pad(xh, [0, 1, 0, 0])[:, :, :, 1:]
        nt = F.pad(xh, [0, 0, 1, 0])[:, :, :-1]
        nb = F.pad(xh, [0, 0, 0, 1])[:, :, 1:]
        n = nl + nr + nt + nb   # simplification of eq 12
        # forward graph
        logit = (h + n).permute(2, 3, 0, 1)
        # backward graph
        h_next = {}
        weight_hh = self.fc_hh.weight.view(self.hidden_channels, self.hidden_channels)
        for i, j, j_ in self.graph_backward(H, W):
            l = logit[i, j]

            if i < H - 1:
                l = l + F.linear(h_next[(i + 1, j)], weight_hh, self.fc_hh.bias)

            if j_ is not None:
                l = l + F.linear(h_next[(i, j_)], weight_hh, self.fc_hh.bias)

            h_next[(i, j)] = torch.sigmoid(l)

        h_next = torch.stack([torch.stack([h_next[(i, j)] for i in range(H)], -1) for j in range(W)], -1)
        return h_next

    def loss_generative(self, x, log_pi, mu, inv_sigma):
        '''
        negative log likelihood, constants are neglected
        x: N, C, H, W
        log_pi: N, K, H, W
        mu: N, KC, H, W
        inv_sigma: N, KCC, H, W
        '''
        d = x.permute(0, 2, 3, 1).unsqueeze(3) - mu
        en = torch.flatten(d.unsqueeze(4) @ inv_sigma @ d.unsqueeze(5), 3)
        lp = inv_sigma.det().abs().log().sub(en).mul(.5).add(log_pi)
        return torch.logsumexp(lp, 3).neg()

    def get_gmm_params(self, h):
        n, _, H, W = h.shape
        g = self.fc_gmm(h)
        pi, mu, s = g.permute(0, 2, 3, 1).split(self.in_split, dim=3)
        log_pi = F.log_softmax(pi, dim=3)
        mu = mu.reshape(n, H, W, self.num_components, self.in_channels)
        s = s.reshape(n, H, W, self.num_components, self.in_channels, self.in_channels)
        inv_sigma = s @ s.transpose(4, 5)  # positive definite, so as its inverse
        return log_pi, mu, inv_sigma

    @staticmethod
    def sample(log_pi, mu, inv_sigma):
        pi = log_pi.exp()
        sigma = inv_sigma.inverse()
        normal = torch.distributions.MultivariateNormal(mu, sigma)
        x = pi.unsqueeze(3) @ normal.sample()
        x = x.squeeze(3)
        return x.clamp(0, 1).permute(0, 3, 1, 2)

    @staticmethod
    def graph_backward(H, W):
        for i in range(H - 1, -1, -1):
            if i % 2:
                for j in range(W):
                    if j:
                        yield i, j, j - 1
                    else:
                        yield i, j, None
            else:
                for j in range(W - 1, -1, -1):
                    if j == W - 1:
                        yield i, j, None
                    else:
                        yield i, j, j + 1
