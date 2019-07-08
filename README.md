# DeepLearning
Deep Learning Course CS4180

## Contents:
- smolnet: Student teacher method
- pytorch-weight-prune-develop: Weight pruning method
- terraform: Automated benchmark setup
- pytorch-AdAIN -master: Evaluated project of Huang et al. 
Also included our life demo -> works by running camera_transfer.py and specifying the style on line 149: style = style_tf(Image.open("input/style/sketch.png"))
The project can be tested by running test.py from the comment line -> example to run an example: python test.py --content_dir input/content --style_dir input/style
The benchmarking code is located in the test.py file.