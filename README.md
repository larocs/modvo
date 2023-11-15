# MODVO

This repository contains the code for MODVO, a modular VO pipeline written in Python. MODVO was designed to make it easy to experiment with different VO modules (such as feature detection and feature matching) and to combine them in different ways. It is also designed to be easy to use and to be easy to integrate with other code.

## Installation

MODVO was tested with Python 3.6 or higher. It can be installed using pip:

```bash
    git clone https://github.com/hudsonmartins/modvo
    cd modvo
    python -m pip install -e .
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