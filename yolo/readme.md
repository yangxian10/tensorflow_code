#YOLO_tensorflow

###YOLO weights file

(1) Train darknet with our data, and get weights file named 'yolo_final.weights'

(2) Convert 'yolo_final.weights' to caffe version by 'create_yolo_caffemodel.py'.

(3) Convert weights to tensorflow style by 'caffe-tensorflow' model and yolo_deploy.prototxt 

###Usage

(1) put 'weights.npy' to current folder.

(2) usage with one image test or images directory

	python yolo.py argvs

	where argvs are

	-w <weight_file>
	-i <img_file>
	-d <img_dir>

###Tips

- tensorflow padding is not suitable for this net, so padding can't be 'SAME', must use tf.pad and 'VALID' !!!
- tensorflow don't have layer of 'Leaky relu', not use 'tf.nn.relu'
- image read model need image rescale to [0,1] and (448,448,3), because darknet model use opencv, so image channel need to change BGR.

###Reference
https://github.com/gliese581gg/YOLO_tensorflow

###Changelog
2016-7-8: First upload!

2016-7-22: change 'np.load().item()' to load parameters