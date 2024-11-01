# MODVO

This repository contains the code for MODVO, a modular VO pipeline written in Python, published in the paper [A comparison of deep learning-based visual odometry algorithms in challenging scenarios](https://ieeexplore.ieee.org/document/10553552). MODVO was designed to make it easy to experiment with different VO modules (such as feature detection and feature matching) and to combine them in different ways. It is also designed to be easy to use and to be easy to integrate with other code. This is a work in progress and we are still working on improving the code and adding more modules.


## Installation

### Using pip
MODVO was tested with Python 3.6 or higher. It can be installed using pip:

```bash
    git clone https://github.com/larocs/modvo
    cd modvo
    python -m pip install -e .
```

Make sure to install the dependencies listed in `requirements.txt`.
Also, initialize the submodules as you need. For example, if you need to use the Superglue matcher, run the following command:

```bash
    git submodule update --init --recursive modvo/thirdparty/SuperGluePretrainedNetwork
```

If you plan to use the GUI to visualize the trajectories, you also need to initialize and install the pangolin submodule:

```bash
    git submodule update --init --recursive modvo/thirdparty/pangolin
```

### Using Docker

If you prefer to use Docker, you can build a Docker image with the following command:

```bash
    docker build -t modvo .
```

## Usage

To run a pipeline with the modules already available in this repository you just need to create a configuration file. The configuration file is a YAML file that specifies the modules to be used and their parameters. For example, the following configuration file runs a pipeline to run a frame-by-frame VO in sequence `03` of KITTI dataset with the ORB feature detector and Brute Force feature matcher:

```yaml
    dataloader:
        class: kitti.KITTILoader
        root_path: '/root/datasets/kitti'
        start_frame: 0
        stop_frame: 800
        sequence_name: '03'
        camera_id: '0'

    detector:
        class: orb.ORBDetector
        nfeatures: 1000
        scaleFactor: 1.2

    matcher:
        class: bf.BFMatcher
        crossCheck: false

    vo:
        class: frame_by_frame.FrameByFrameTracker
```

## Adding a new module

To add a new module to MODVO you need to create a new Python file inside the `modvo/<module_type>` folder, where the `module_type` is the type of module you want to add (e.g. detector, matcher, etc.). The python files must contain a class that inherits from modvo.<module_type>.<ModuleType> and implements the methods defined in the parent class. The `__init__` method of the class must receive a dictionary with the parameters of the module. For example, the following code shows the implementation of the ORB feature detector:

```python
    import cv2
    import numpy as np
    from modvo.detectors.detector import Detector

    class ORBDetector(Detector):  
    def __init__(self, **params):
        self.detector = cv2.ORB_create(**params)

    def getNLevels(self):
        return self.detector.getNLevels()

    def getScaleFactor(self):
        return self.detector.getScaleFactor()
    
    def detectAndCompute(self, image):
        [kpts_cv, descriptors] = self.detector.detectAndCompute(image, None)
        keypoints = tools.convert_kpts_cv_to_numpy(kpts_cv)
        octaves = np.array([kpt.octave for kpt in kpts_cv])
        scores = np.array([kpt.response for kpt in kpts_cv])

        self.features = {'keypoints': keypoints,
                         'octaves'   : octaves,
                         'descriptors': descriptors,
                         'scores': scores}
        return self.features
```
## Citation
If you find our work useful, please consider citing our paper:
```
@INPROCEEDINGS{10553552,
  author={Bruno, Hudson M. S. and Cabral, Kleber M. and Colombini, Esther L. and Givigi, Sidney N.},
  booktitle={2024 IEEE International Systems Conference (SysCon)}, 
  title={A comparison of deep learning-based visual odometry algorithms in challenging scenarios}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  keywords={Deep learning;Visualization;Accuracy;Feature detection;Pipelines;Feature extraction;Cameras},
  doi={10.1109/SysCon61195.2024.10553552}}
```
## Dataset
The dataset used in the paper above is available [here](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/VIG3FC).

## Other publications
Also checkout our RGB-D Visual SLAM dataset with emulated camera failures [here](https://larocs.github.io/queenscamp-dataset/).

## References and Inspiration

This repository was inspired by the following works:

[ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)

[pyslam](https://github.com/luigifreda/pyslam)

[Python-VO](https://github.com/Shiaoming/Python-VO)

[evo](https://github.com/MichaelGrupp/evo)

[RTAB-Map](https://github.com/introlab/rtabmap)
