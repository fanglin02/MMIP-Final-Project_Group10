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

Note: To visualize intermediate results for each processing step (e.g., semantic masks, depth maps, layered RGBA images), run the scripts located in the respective folders: [2D to 3D](2d_to_3d/) and [Parallax Scrolling](parallax_crolling/).

## Results 
### 2D-to-3D Pop-Out Rendering
The demonstration videos are available in [Parallax Scrolling Demo](https://github.com/user-attachments/assets/6642abeb-981a-4b7b-9ba8-5b71dec42d78) 
This demo illustrates how a single 2D image is converted into layered pseudo-3D representation, showing smooth depth-aware motion of foreground objects.

### Parallax Scrolling
The demonstration videos are available in [Parallax Scrolling Demo](https://github.com/user-attachments/assets/bdfd11a0-c896-495d-818e-fa27c1df817f).
This demo shows depth-guided scene animations with foreground, midground, and background layers moving at different speeds to produce a convincing parallax effect.

## Ablation Study
1. Seamless Tiling
The ablation study videos are available in: [Mirroring](https://github.com/user-attachments/assets/1e0ff693-2fce-4384-8cb6-60c98b16f1ce) and [Without Mirroring](https://github.com/user-attachments/assets/8145642d-470a-4d31-92aa-4c49156296da).
Demonstrates the importance of mirroring and concatenating layers for continuous horizontal motion. Without mirroring, edges appear misaligned, causing visible jumps in the animation.
2. Support Layer
The ablation study videos are available in: [Support Layer](https://github.com/user-attachments/assets/49fa266a-a613-4c57-9f70-2c6e625f5e2c).
Shows the effect of combining foreground and midground layers into a support layer. Without this step, gaps or holes appear in the midground, breaking the visual continuity.

## Report
The implementation details, methodology, and results are documented in [Group10_report.pdf](Group10_report.pdf)