# how to 
> \$ git clone --recurse-submodules git@github.com:KaleidoZhouYN/multi_input_face_recognition.git

## test
1.download r100 model checkpoint to multi_input/work_dirs/ms1m_r100:

https://drive.google.com/file/d/1JSfz28fAE8kDNg4L_PgF9wSLL5JUKJ9Z/view?usp=sharing

2.prepare 3 different aligned face image of 1 person 

3.cp 3 image into ./demo/p1 and rename 0-3.jpg

4.do the same thing to another person to ./demo/p2

> \$ cd multi_input<br>
> \$ python inference.py

## train
to do 