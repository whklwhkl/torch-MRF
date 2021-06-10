import sys
import torch

from PIL import Image
from src.mrf import MRF
from torchvision.transforms import ToTensor
from tqdm import trange


model = MRF(3, 128, 8)
im = Image.open(sys.argv[1])
im = im.convert('RGB')
tran = ToTensor()
x = tran(im)[None]
# test
h = torch.zeros(1, model.hidden_channels, *x.shape[2:])

min = 0
sgd = torch.optim.RMSprop(model.parameters(), lr=3e-4)
with trange(100) as pbar:
    for i in pbar:
        sgd.zero_grad()
        h = model(x)
        gmm = model.get_gmm_params(h)
        l = model.loss_generative(x, *gmm).mean()
        l.backward()
        sgd.step()
        l = l.item()
        pbar.desc = f'loss={l:.4f}'
        if l < min:
            min = l
            torch.save(model, 'model.pth')
