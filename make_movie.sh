#!/bin/bash
avconv -r 1000 -i weights/weight_%d.png -b:v 10000k test6.mp4

