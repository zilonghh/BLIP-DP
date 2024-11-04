# BLIP-DP：Dynamic Text Prompt Joint Multimodal Features for Accurate Plant Disease Image Captioning

The complete dataset will be published after the paper is accepted.

The example dataset is reflected in data/example.

# Install

 The code has been tested on PyTorch 1.10.
To install the dependencies, run <pre/>pip install -r requirements.txt</pre>

# Model
Please download the training weights for the image captioning model and VQA model from the link below
<a href="https://pan.baidu.com/s/1wy-S3GHJfM2Vcwnuse1Rlw?pwd=h5nf">Models</a>

# Image Captioning:

To train on our dataset, the code should be run:

python -m torch.distributed.run --nproc_per_node=5 train_caption.py
