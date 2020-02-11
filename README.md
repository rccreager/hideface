# HideFace 

Please see `http://bit.ly/rccreager_slides` for the slides corresponding to this package!

HideFace is a testing framework for face detectors and adversarial attacks.
Generally, we can divide image attacks into two classes:
* White-box: attacks that use information about the algorithm being attacked (for example, model weights)
* Black-box: attacks that assume no knowledge of the algorithm being attacked

This package is designed for **black-box** attacks against facial detection algorithms because no information about the model itself is used. 
This framework is useful for adding privacy filters to images (to hide them from face detection) or for testing the robustness of face detection algorithms against attacks.

Attacks are implemented in the `hideface/attacks.py` file -- a few examples are provided.
Detectors are provided as a dictionary in the `run_attacks.py` script -- for now, only the DLib HoG detector is used. 

## Setup and Installation
This package uses DLib for its face detection -- this is a C++ package and it takes a long time to build, so be patient! 
    
    git clone https://github.com/rccreager/hideface.git
    cd hideface
    source built/apt_get.sh
    python3 -m venv build/hideface_env
    source build/hideface_env/bin/activate
    pip3 install -r build/requirements.txt

## Running a Test Attack  

For a test and example of how this package works, run `example_attack.py`, which will test 100 images from a subset of the WIDER-FACE dataset:

    python3 example_attack.py
