#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Script used for obtaining the calibration matrix of a camera setup.

Put verbose descriptions here.

Example
-------

TODO
----
- Handle errors if cannot find checkerboard within single image
"""

def main():
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    import time
    import cv2
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    rawCapture = PiRGBArray(camera)
    # allow the camera to warmup
    time.sleep(0.1)
    # grab an image from the camera
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
    # display the image on screen and wait for a keypress
    cv2.imshow("Image", image)
    cv2.waitKey(0)
