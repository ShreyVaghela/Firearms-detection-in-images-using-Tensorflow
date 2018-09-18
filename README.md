# Firearms-detection-in-images-using-Tensorflow


Object Detection is the process of finding the real world entities such as human,bicycle,cars etc. Object detection algorithm uses extracted features in order to detect these entities in image and then runs the classifier in order to classify them in one of these several entities. Custom object can be detected by training the model on your custom object with the information about the position of your object in images and class label. 

### Tensorflow API

You can download the tensorflow api from [here](https://github.com/tensorflow/models/tree/master/research/object_detection).
### Dataset

The project detects firearms like pistol in images using Tensorflow. It was made by using the images of pistols provided [here](https://sci2s.ugr.es/weapons-detection). 

### LabelImg

[LabelImg](https://github.com/tzutalin/labelImg) has been used in order to annotate the images. The annotation are saved as XML files in PASCAL VOC format as that used in ImageNet. Satisfying the required dependencies for your respective machines it will pop an interface where you need to select the location of your above downloaded dataset. Figure shows the example of drawing bounding box and labeling the image.
![screenshot 12](https://user-images.githubusercontent.com/20052459/45665781-35309200-bb30-11e8-81e3-4290f5415752.png)

### XML to CSV and generating tensor records

After generating xml file using labelImg we need to convert those files into csv in order to generate tensor records which could be further provided to the network to train the model. Find the xml_to_csv.py and generate_tfrecords.py

### Train the model
Run the following command from the root directory. Replace the `pipeline.config` with the config file of the pre trained model that you are using for training.
`python object_detection/train.py \
        --logtostderr \
        --train_dir=train \
        --pipeline_config_path=pipeline.config`
        
After training move the checkpoints with highest step number to root directory. It should contain 3 files:
`model.ckpt-STEP_NUMBER.data-00000-of-00001`
`model.ckpt-STEP_NUMBER.index`
`model.ckpt-STEP_NUMBER.meta`

### Generating frozen_inference_graph

Generate inference graph by:
`python object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path faster_rcnn_resnet101.config \
        --trained_checkpoint_prefix model.ckpt-STEP_NUMBER \
        --output_directory output_inference_graph`
### Testing the model
Run the `object_detection_tutorial.ipynb` in the root directory by specifying the path to the `frozen_inference_graph.pb`, `label_map.pbtxt`, `test_images`.

### Output
![download 3](https://user-images.githubusercontent.com/20052459/45665884-b0924380-bb30-11e8-99b2-574abbfcbaff.png)
![download 2](https://user-images.githubusercontent.com/20052459/45665898-c99af480-bb30-11e8-9d5c-8701ee27d891.png)
![download 1](https://user-images.githubusercontent.com/20052459/45665908-dae40100-bb30-11e8-9576-02afbf1c8928.png)

### Future Scope
Currently, the project only detects pistols in images. Further it can be extended to detect firearms in videos as well as in real time. An alarming system can be made by triggering an action like message to the nearby police station when a pistol is detected in the security cameras.

### Refernces
[Tensorflow API](https://github.com/tensorflow/models/tree/master/research/object_detection)

[Raccoon Detector Dataset](https://github.com/datitran/raccoon_dataset)

[sentedx](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku)


