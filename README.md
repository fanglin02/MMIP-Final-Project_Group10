# 2D-to-3D and Parallax Scrolling : A Depth-Based Multimodal Image Layering System

## Description
This project implements a depth-based multimodal image layering system that generates dynamic, depth-aware visual effects from a single 2D image. The system consists of two independent modules:
1. **2D-to-3D Pop-Out Rendering**
2. **Parallax Scrolling**

## Installation
1. Clone the repository:
```bash
git clone https://github.com/fanglin02/MMIP-Final-Project_Group10.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. 2D-to-3D Pop-Out Rendering
```bash
python 2dto3d.py
```

2. Parallax Scrolling
```bash
python parallax_scrolling.py
```
Output files will be saved in the corresponding folders inside [output](output/).

Note: To visualize intermediate results for each processing step (e.g., semantic masks, depth maps, layered RGBA images), run the scripts located in the respective folders: [2D to 3D](2d_to_3d/) and [Parallax Scrolling](parallax_crolling/).

## Results and Ablation Study
All demonstration videos can be found in the following directory: [Demo](demo/).
All ablation study videos are available in: [Ablation Study](ablation_study/).

## Report
The implementation details, methodology, and results are documented in [Group10_report.pdf](Group10_report.pdf)