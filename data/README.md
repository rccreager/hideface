## Downloading testing data

To download test data and labels:

    apt-get install unzip

    pip3 install gdown

    gdown https://drive.google.com/uc?id=0B6eKvaijfFUDQUUwd21EckhUbWs&export=download

    unzip WIDER_train.zip

    wget "http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip"

    unzip wider_face_split.zip

## Downloading a model

    wget "http://dlib.net/files/mmod_human_face_detector.dat.bz2"

    bzip2 -dk mmod_human_face_detector.dat.bz2
