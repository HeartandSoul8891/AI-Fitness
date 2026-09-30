==============================
= YOLO Trainer Module Manual =
==============================
Overview
The YOLO Trainer module provides a graphical interface for training Ultralytics YOLO models using either:

Bounding Box Detection (BBox)
Image Segmentation (Segm)
The trainer allows users to configure datasets, model architectures, training hyperparameters, augmentation strategies, and output settings without manually writing training scripts.

Supported Training Modes
Bounding Box (BBox)
Used for traditional object detection tasks where objects are annotated with rectangular bounding boxes.

Examples:

Vehicle detection
Person detection
Wildlife monitoring
Defect detection
Supported default models:

yolo26n.pt
yolo26s.pt
yolo26m.pt
yolo26l.pt
yolo26x.pt
Segmentation (Segm)
Used for pixel-level object segmentation.

Examples:

Medical imaging
Building extraction
Instance segmentation
Agricultural analysis
Supported default models:

yolo26n-seg.pt
yolo26s-seg.pt
yolo26m-seg.pt
yolo26l-seg.pt
yolo26x-seg.pt
Interface Layout
The interface is divided into four sections:

Dataset Selection
Model Architecture & Setup
Advanced Settings & Validation
Execute Training
1. Dataset Selection
Automatic Dataset Discovery
The trainer automatically scans the configured dataset directory:




Plain Text
training/
including all subfolders for:




Plain Text
data.yaml
files.

Any discovered dataset configuration file will appear in the dataset selection list.

Manual Dataset Selection
If no dataset is found, a manual path can be entered.

Example:




Plain Text
training/my_dataset/data.yaml
The selected file must exist before training can start.

Dataset Structure Example



Plain Text
training/
└── my_dataset/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
2. Model Architecture & Setup
Model Weights / Architecture
Select the base model used for training.

The trainer searches the configured model directory for local .pt files.

Example:




Plain Text
ultralytics/bbox/
or




Plain Text
ultralytics/segm/
If no local models are available, default YOLO models can be selected.

Epochs
Controls how many complete passes are made through the training dataset.

Typical values
Dataset Size	Recommended Epochs
Small	50-150
Medium	100-300
Large	200-500
Higher values may improve accuracy but increase training time.

Image Size
Defines the image resolution used during training.

Available options:

640
960
1024
1280
Recommendations
Use Case	Resolution
Fast training	640
Balanced	960
High detail	1024-1280
Higher resolutions require more VRAM.

Batch Size
Number of images processed simultaneously.

Options:

2
4
8
16
32
Notes
Larger values:

Train faster
Require more GPU memory
Smaller values:

Reduce VRAM consumption
Increase training duration
Device ID
Specifies the processing device.

Examples:

GPU 0




Plain Text
0
GPU 1




Plain Text
1
CPU




Plain Text
cpu
Output Directory
Defines where training runs are stored.

Example:




Plain Text
output/ultralytics/bbox
Each training session generates a separate experiment folder.

Run Name
Custom name for the training session.

Example:




Plain Text
vehicle_detector_v1
Resulting folder:




Plain Text
output/ultralytics/bbox/vehicle_detector_v1
If left empty, the trainer uses:




Plain Text
exp
Single Class Mode
Treats all classes as a single category.

Useful when:

Multiple labels should behave as one class
Binary object detection is desired
Example:




Plain Text
car
truck
bus
becomes:




Plain Text
vehicle
3. Advanced Settings & Validation
Freeze Layers
Prevents part of the neural network from updating during training.

Benefits
Faster training
Reduced overfitting
Better transfer learning performance
Typical values:




Plain Text
0 = train everything
10 = partially frozen
20+ = heavily frozen
Weight Decay
Regularization strength.

Default:




Plain Text
0.0005
Purpose:

Reduces overfitting
Improves generalization
Increase cautiously.

Patience
Early stopping parameter.

Example:




Plain Text
15
Training stops if validation performance does not improve for 15 epochs.

Benefits:

Saves training time
Prevents overtraining
Confidence Threshold
Minimum confidence required during:

Validation
Inference
Default:




Plain Text
0.35
Higher values:

Fewer detections
More reliable detections
Lower values:

More detections
More false positives
Data Augmentation Settings
The trainer supports on-the-fly augmentation.

Augmentations are generated during training and do not modify original images.

Color (HSV)
HSV-Hue
Controls color shifts.

Typical:




Plain Text
0.015
HSV-Saturation
Controls color intensity variation.

Default:




Plain Text
0.7
HSV-Value
Controls brightness variation.

Default:




Plain Text
0.4
Geometric & Scale
Rotation
Random image rotation.

Range:




Plain Text
0° - 180°
Translate
Random image shifting.

Range:




Plain Text
0.0 - 1.0
Scale Gain
Random zoom in/out.

Range:




Plain Text
0.0 - 1.0
Horizontal Flip Probability
Chance of horizontal mirroring.

Typical:




Plain Text
0.5
Vertical Flip Probability
Chance of vertical mirroring.

Default:




Plain Text
0.0
Recommended only when object orientation is irrelevant.

Composition & Erasure
Mosaic
Combines multiple images into one training sample.

Default:




Plain Text
1.0
Benefits:

Better generalization
Improved small object detection
MixUp
Blends images together.

Default:




Plain Text
0.0
Can improve robustness for difficult datasets.

Random Erasing
Randomly removes image areas during training.

Default:




Plain Text
0.4
Benefits:

Simulates occlusions
Improves robustness
Close Mosaic
Number of final epochs that disable Mosaic augmentation.

Default:




Plain Text
10
This often stabilizes model convergence near training completion.

4. Execute Training
Starting Training
Press:




Plain Text
Start Training
The trainer validates:

Dataset path exists
data.yaml file exists
Model selection is valid
If validation succeeds, training begins.

Training Progress
During execution the interface displays:




Plain Text
Training model in progress...
Training is performed using the configured Ultralytics settings.

Completion
On successful completion, a message similar to the following appears:




Plain Text
Training completed!
Results saved to:
 
output/ultralytics/bbox/my_run
Output Files
Typical YOLO output structure:




Plain Text
output/
└── ultralytics/
    └── bbox/
        └── my_run/
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            ├── results.png
            ├── confusion_matrix.png
            ├── PR_curve.png
            └── args.yaml
Best Practices
Small Datasets
Enable Mosaic
Use strong augmentation
Train 100-300 epochs
Large Datasets
Reduce heavy augmentation
Use larger batch sizes
Consider larger model variants
Limited VRAM
Use:




Plain Text
Image Size: 640
Batch Size: 2 or 4
Model: yolo26n
Maximum Accuracy
Use:




Plain Text
Image Size: 1024-1280
Model: yolo26l or yolo26x
Longer training runs
Troubleshooting
No Dataset Found
Ensure a valid:




Plain Text
data.yaml
exists within the training directory.

Out of Memory (CUDA OOM)
Reduce:

Image Size
Batch Size
or use a smaller model.

Training Does Not Start
Verify:

Dataset path exists
data.yaml is valid
Model file exists
GPU is available (if selected)
Poor Accuracy
Try:

More training epochs
Better annotations
Increased dataset size
Adjust augmentation settings
Use a larger YOLO model
Summary
The YOLO Trainer module provides a complete GUI-based workflow for training Ultralytics YOLO detection and segmentation models. It combines automatic dataset discovery, model management, advanced augmentation controls, validation settings, and experiment tracking into a single interface, making it suitable for both beginners and advanced users developing custom computer vision models.


