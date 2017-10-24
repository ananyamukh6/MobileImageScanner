I pledge on my honor that I have not received any unauthorized assistance on this project and that I have properly acknowledged all sources of material that was not solely produced by me."
Electronically signed by Ananya Mukherjee Date: 5/16/17

The main files are:
1. mobilescanner_final_curr.py - Main file
2. basicscan.py - Run this file to see the basic scan output. I have set display=False in mobilescanner_final_curr.py (line 14). If we set display=True in mobilescanner_final_curr.py (line 14) we can see the intermediate step outputs.
3. mobilescanner_formfilling.py - Contains code for reading filled forms. To read regions of text from a form. When running this file if the 
error "UnicodeEncodeError: 'charmap' codec can't encode characters in position 87-88: character maps to <undefined>" is encountered please run'chcp 65001' on terminal 
4. mobilescanner_partialform.py - Contains code for joining partial forms. This file imports panorama.py which is adapted from https://compvisionlab.wordpress.com/2013/05/13/image-blending-using-pyramid/
5. panorama.py


