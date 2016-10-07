# Weakly_detector
Tensorflow implementation of "Learning Deep Features for Discriminative Localization"

B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba
Learning Deep Features for Discriminative Localization.
Computer Vision and Pattern Recognition (CVPR), 2016.
[[PDF](http://arxiv.org/pdf/1512.04150.pdf)][[Project Page](http://cnnlocalization.csail.mit.edu/)]

# Downloads
In order to run the code you'll need weights initialisation (Best thing is to BRIBE Marc for the files :D)
Otherwise, download some files at https://drive.google.com/drive/folders/0B8tQbRXEAWraNnhJb21DNlNtTG8?usp=sharing and put them in your folder architecture as follows :
1. ./caffe_layers_value.pickle
2. ./models/PERSO/PERSO.VGG16_CAM_W_S.rmsProp.1e-5/model-24
3. ./models/PERSO/PERSO.VGG16_CAM_W_S.rmsProp.1e-5/model-24.meta

Also download the git repository :
https://github.com/marc-moreaux/cm_perso <br/>
and :<br/>
echo "<br/>
<br/>
PYTHONPATH=\$PYTHONPATH:~/work/cm_perso/py/utils<br/>
export PYTHONPATH" >> ~/.bashrc<br/>
