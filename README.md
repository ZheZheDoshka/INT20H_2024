# INT20H 2024
 
Repository for INT20H 2024 team project of Криптомонахи team

Task description is available [here](case_hakaton.pdf) \
Dataset for the project is available [here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar)

### File structure:
- models - used models
    - haarcascade_frontalface_default.xml
    - haarcascade_profileface.xml
    - shape_predictor_68_face_landmarks.dat
- data
    - wiki_crop - dataset
- notebooks - jupyter notebooks for data processing

### Used models:
- [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) - dlib model for face landmarks detection
- [haarcascade_frontalface_default.xml](models%2Fhaarcascade_frontalface_default.xml) - basic OpenCV model for frontal face detection
- [haarcascade_profileface.xml](models%2Fhaarcascade_profileface.xml) - basic OpenCV model for profile face detection
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - model for enhancing the quality of the average faces

### Notebooks:
Presented in the order of execution
- image_filtering.ipynb - initial filtering of the dataset to reduce the number of images
- [processing_opencv_pose_detection.ipynb](notebooks%2Fprocessing_opencv_pose_detection.ipynb) - processing the dataset with basic OpenCV pose detection
- [processing_mtcnn_pose_detection.ipynb](notebooks%2Fprocessing_mtcnn_pose_detection.ipynb) - processing the dataset with advanced MTCNN pose detection based on angle between eyes and nose
- [processing_merge.ipynb](notebooks%2Fprocessing_merge.ipynb) - merging the results of initial filtering and MTCNN pose detection
- [processing_mirror_by_angle.ipynb](notebooks%2Fprocessing_mirror_by_angle.ipynb) - mirroring the images based on the angle between eyes and nose to unify the dataset
- [load_dataset.ipynb](notebooks%2Fload_dataset.ipynb) - loading the dataset for clustering
- ...
- [morphing_preparation.ipynb](notebooks%2Fmorphing_preparation.ipynb) - preparing the clusters for morphing
- [morphing_run.ipynb](notebooks%2Fmorphing_run.ipynb) - running the morphing to create average faces for each cluster
