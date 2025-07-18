# Offically Supported TensorFlow 2.1+ Models on Cloud TPU

## Natural Language Processing

*   [bert](https://arxiv.org/abs/1810.04805): A powerful pre-trained language representation model:
    BERT, which stands for Bidirectional Encoder Representations from
    Transformers.
    [BERT FineTuning with Cloud TPU](https://cloud.google.com/ai-platform/training/docs/algorithms/bert-start) provides step by step instructions on Cloud TPU training. You can look [Bert MNLI Tensorboard.dev metrics](https://tensorboard.dev/experiment/LijZ1IrERxKALQfr76gndA) for MNLI fine tuning task.
*   [transformer](nlp/transformer): A transformer model to translate the WMT
    English to German dataset.
        [Training transformer on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/transformer-2.x) for step by step instructions on Cloud TPU training.

## Computer Vision

*   [efficientnet](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/efficientnet.py): A family of convolutional
    neural networks that scale by balancing network depth, width, and
    resolution and can be used to classify ImageNet's dataset of 1000 classes.
    See [Tensorboard.dev training metrics](https://tensorboard.dev/experiment/KnaWjrq5TXGfv0NW5m7rpg/#scalars).
*   [mnist](https://www.tensorflow.org/datasets/catalog/mnist): A basic model to classify digits
    from the MNIST dataset. See [Running MNIST on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/mnist-2.x) tutorial and [Tensorboard.dev metrics](https://tensorboard.dev/experiment/mIah5lppTASvrHqWrdr6NA).
*   [mask-rcnn](https://www.tensorflow.org/api_docs/python/tfm/vision/configs/maskrcnn/MaskRCNN): An object detection and instance segmentation model. See [Tensorboard.dev training metrics](https://tensorboard.dev/experiment/LH7k0fMsRwqUAcE09o9kPA).
*   [resnet]((https://www.tensorflow.org/api_docs/python/tfm/vision/configs/image_classification/image_classification_imagenet)): A deep residual network that can
    be used to classify ImageNet's dataset of 1000 classes.
    See [Training ResNet on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/resnet-2.x) tutorial and [Tensorboard.dev metrics](https://tensorboard.dev/experiment/CxlDK8YMRrSpYEGtBRpOhg).
*   [retinanet](https://www.tensorflow.org/api_docs/python/tfm/vision/retinanet): A fast and powerful object detector. See [Tensorboard.dev training metrics](https://tensorboard.dev/experiment/b8NRnWU3TqG6Rw0UxueU6Q).
*   [shapemask](https://cloud.google.com/tpu/docs/tutorials/shapemask-2.x): An object detection and instance segmentation model using shape priors. See [Tensorboard.dev training metrics](https://tensorboard.dev/experiment/ZbXgVoc6Rf6mBRlPj0JpLA).

## Recommendation
*   [dlrm](recommendation/ranking): [Deep Learning Recommendation Model for
Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091).
*   [dcn v2](recommendation/ranking): [Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535).
*   [ncf](recommendation): Neural Collaborative Filtering. See [Tensorboard.dev training metrics](https://tensorboard.dev/experiment/0k3gKjZlR1ewkVTRyLB6IQ).
