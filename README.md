# Face Identification

This example is based on [VGGFace](https://github.com/rcmalli/keras-vggface) model to get a person face features and store as pre-trained features. 
Real-time image then compared with every pre-computed label features. I suspect, it won't work
fast on thousands faces due to O(n) complexity, where `n` is a number of labels.

## Scala Version

Arcticle descring the approach to identify faces: [Face Identification with VGGFace and OpenCV](https://novakov-alexey.github.io/face-identification/)

See modules here:

- [src/main/scala](src/main/scala)

## Python Version
See main two scripts for details:

-   [src/extract_features.py](src/extract_features.py)
-   [src/face_identify_demo.py](src/face_identify_demo.py)
