# README #


### What is this repository for? ###

* Mobile Image scanner with geometric and appearance correction using OpenCV


### Dependencies ###

This code runs on Python 2. It also uses libraries like sklearn, skimage, OpenCV

### How to run the code###

* mobilescanner_final_curr.py - Main file
* basicscan.py - Run this file to see the basic scan output. I have set display=False in mobilescanner_final_curr.py (line 14). If we set display=True in mobilescanner_final_curr.py (line 14) we can see the intermediate step outputs.
*  mobilescanner_formfilling.py - Contains code for reading filled forms. To read regions of text from a form. When running this file if the 
error "UnicodeEncodeError: 'charmap' codec can't encode characters in position 87-88: character maps to <undefined>" is encountered please run'chcp 65001' on terminal 
*  mobilescanner_partialform.py - Contains code for joining partial forms. This file imports panorama.py which is adapted from https://compvisionlab.wordpress.com/2013/05/13/image-blending-using-pyramid/
* panorama.py - Contains code for image blending

### Detailed Analysis ###
For more details check out below files under src: 
Detailed report: Ananya Mukherjee Final.pdf
Slides: Mobile Image Scanner.pptx