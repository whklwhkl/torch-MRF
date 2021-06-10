# torch-MRF
this repo implements [Deep Markov Random Field for Image Modeling](https://arxiv.org/abs/1609.02036) in *PyTorch*


## Usage
1. train with your favourite image
```bash
python scripts/train.py ${IMAGE_PATH}
```

2. generate textures using the trained model
```bash
python scripts/demo.py
```

3. check the generated image `demo.png`
