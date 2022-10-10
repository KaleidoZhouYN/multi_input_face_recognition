# how to 
> \$ git clone --recurse-submodules git@github.com:KaleidoZhouYN/multi_input_face_recognition.git

## test

1. download r100 model checkpoint to multi_input/work_dirs/ms1m_r100:

    https://drive.google.com/file/d/1JSfz28fAE8kDNg4L_PgF9wSLL5JUKJ9Z/view?usp=sharing

2. prepare 3 different aligned face image of 1 person 

3. cp 3 image into ./demo/p1 and rename 0-3.jpg

4. do the same thing to another person to ./demo/p2

5. run

    > \$ cd multi_input<br>
    > \$ python inference.py

## train
1. download ms1m_v3 dataset from insightface
2. copy ms1m_v3 dataset into /dev/shm
3. run 
    > \$ cd multi_input<br>
    > \$ bash run_ms.sh


# example

similarity between p1 & p2(same person)

|   |![p1_0](./demo/p1/0.jpg)|![p1_1](./demo/p1/1.jpg)|![p1_2](./demo/p1/2.jpg)|
|:-:|:-:|:-:|:-:|
|![p2_0](./demo/p2/0.jpg)| | | |
|![p2_1](./demo/p2/1.jpg)| | | |
|![p2_2](./demo/p2/2.jpg)| | | |