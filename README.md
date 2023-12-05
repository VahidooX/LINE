# LINE: Large-scale information network embedding

This is a Python implementation of LINE (Large-scale information network embedding). It needs Theano and Keras libraries to get executed.
LINE is a node embedding algorithm for networks. It is initially proposed in the following paper:

Tang, Jian, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei. "[Line: Large-scale information network embedding.](http://www.www2015.it/documents/proceedings/proceedings/p1067.pdf)" In Proceedings of the 24th International Conference on World Wide Web, 2015.

You may find the original implementation of LINE in C++ on [Github](https://github.com/tangjianpku/LINE).

# Running instructions with docker (Using GPU)
1. docker pull tensorflow/tensorflow:latest-gpu-jupyter TAG id: 2de4ac3c6e83 Size: 3.56 GB
2. pull the repository
3. Run main.py

Test if gpu is available inside container:
gpu_available = tf.test.is_gpu_available()

Notes: 
To use GPU inside docker, we need nvidia-container-toolkit (for latest versions of docker) # https://www.tensorflow.org/api_docs/python/tf/test/is_gpu_available

Docker instructions tensorflow: https://www.tensorflow.org/install/docker

If you need running instructions for the original version of the code (written in cpp), please get in touch with me by raising an issue.
