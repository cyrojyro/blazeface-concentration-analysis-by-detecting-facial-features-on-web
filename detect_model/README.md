## Blazeface Implementation

This directory contains Blazeface Implementation & simple tfjs model convert script.

### Results

Please check out sample image directory before execution.

```
python test.py --model ./pretrained.hdf5 --threshold 0.5 --tie_threshold 0.2
```

### Scripts

- train.py: training script, check out train_config.ini
- test.py: testing script
- camera.py: test model with your camera
- convert_to_tfjs.py: converts .hdf5 file to tensorflowjs model
- generate_anchors.py: generates anchor

### Convert to TFJS model

```
    sh convert_to_tfjs.sh [.hdf5 file] [output directory]
```

### References

##### Model & Training Methods

- [BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs, Valentin Bazarevsky et al. 19-07](https://arxiv.org/pdf/1907.05047.pdf)
- [SSD: Single Shot MultiBox Detector, Wei Liu et al. 15-12](https://arxiv.org/pdf/1512.02325.pdf)

### Datasets

- [WIDERface](http://shuoyang1213.me/WIDERFACE/)
- [FDDB](http://vis-www.cs.umass.edu/fddb/)
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
