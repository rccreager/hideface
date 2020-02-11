# HideFace 

Please see `http://bit.ly/rccreager_slides` for the slides corresponding to this package!

HideFace is a testing framework for face detectors and black box attacks.
This framework is useful for adding privacy filters to images (to hide them from face detection) or for testing the robustness of face detection algorithms against attacks.

Attacks are implemented in the `hideface/attacks.py` file -- a few examples are provided.
Detectors are provided as a list in the `run_attacks.py` script -- for now, only the DLib HoG detector is used. 

## Setup and Installation

    git clone https://github.com/rccreager/hideface.git
    source built/apt_get.sh
    python3 -m venv build/hideface_env
    source build/hideface_env/bin/activate
    pip3 install -r build/requirements.txt

## Running a Test Attack  



