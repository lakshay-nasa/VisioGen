<h1 align="center">3d Reconstruction: VisioGen</h1>

<p align="center">
<img src="https://raw.githubusercontent.com/lakshay-nasa/VisioGen/main/.github/images/drdo.png" width="200" height="">
</p>

## 🗒️ Description:
<p>In my research endeavour at IRDE, DRDO, Dehradun. I dedicated myself to advancing the processing of visual data through an innovative approach: converting 2D images into intricate 3D models. Leveraging a combination of image processing techniques with Structure From Motion Technique and libraries like:</p>

<ul><li>lightglue</li><li>OpenCV</li><li>PyTorch</li><li>SciPy</li><li>pyvista</li><li>MatplotLib</li><li>Joblib</li></ul>

<p>and using computer vision libraries as well as methodologies like Triangulation and TKf algorithm. With goal to open new doors in spatial modeling applications ranging from automation and gaming to robotics, architecture, and beyond.</p>

**Developed the project as a part of my Internship at DRDO (Defence Research and Development Organisation)'s Dehradun headquarters.**



<img src="https://raw.githubusercontent.com/lakshay-nasa/VisioGen/main/.github/images/fountain2D.png" alt="fountain2D" width="50%" />
➤
<img src="https://raw.githubusercontent.com/lakshay-nasa/VisioGen/main/.github/images/VisioGen.gif" alt="fountain3D" width="50%" />


## 📽 Sample Demo:
https://www.youtube.com/watch?v=RcAH8HbFBMI


## How to Use

1. Make sure Python is installed on your system. If not, you can download it from [Python's official website](https://www.python.org/downloads/).
2. Install and create a virtual environment using the following command: `python -m venv .venv`

3. Ensure you have the necessary dependencies installed by utilizing `requirements.txt`. You can install them with: `pip install -r requirements.txt`
4. Start the GUI application: `python gui.py`
5. Input images from any dataset.
6. Set the desired parameters (optional).
7. Click "Start Process" to begin the reconstruction.
8. To cancel the process, click "Cancel Process" and wait until the "process cancelled" message appears in the log.

**Caution:** Avoid clicking "Start Process" immediately after pressing the "Cancel Process" button to prevent unwanted background resource consumption and potential software crashes.

## Notes

1. The dataset containing large number of input visuals. Processing it will take a significant amount of time depending on your PC specifications.
2. It is recommended to use other datasets with an increased value of `max num keypoints`. You can use `4096` or `8192` in place of the default value of `2048`, but be aware that it will take more time to process.
3. Change parameters at your own risk, depending on your specific system specifications.

## 🎓 DRDO (Defence Research and Development Organisation) Internship Certificate:
<p align="center">
<img src="https://raw.githubusercontent.com/lakshay-nasa/VisioGen/main/.github/images/Certificate_DRDO_Lakshay.png" width="600" height="">
</p>

## 📝 Internship/ Project Final Report link:
https://drive.google.com/file/d/1NVCMCRxbT-sStr9ozld1Yu5m53JrQngQ/view?usp=sharing


##  🫙 Databases ->
- https://phototour.cs.washington.edu/
- https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/denseMVS.html

## License

This project is licensed under the **Educational and Non-Commercial License** - see the [LICENSE.md](LICENSE.md) file for details.
