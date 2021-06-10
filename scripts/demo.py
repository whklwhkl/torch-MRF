import torch
import torch.nn.functional as F

from torchvision.transforms import ToPILImage


model = torch.load('model.pth')

x = torch.rand(1, 3, 1, 1)
h = None

for i in range(32):
    x = F.pad(x, [1] * 4)
    h = model(x)
    g = model.get_gmm_params(h)
    x = model.sample(*g)
    ToPILImage()(x[0]).save(f'demo.png')
