# Joint Action Unit localisation and intensity estimation

This is a built-in class for Action Unit intensity estimation with heatmap regression, adapted from the code used for the BMVC paper "Joint Action Unit localisation and intensity estimation through heatmap regression" (see citation below)

![Alt Text](https://esanchezlozano.github.io/files/animated_heatmaps_AU6.gif)

![Alt Text](https://esanchezlozano.github.io/files/animated_heatmaps_AU12new.gif)

This class takes an image and returns the heatmaps and the AU predictions from them. In order to create a standalone class the points are detected using the dlib facial landmark detector. This will be shortly replaced by the [iCCR](http://esanchezlozano.github.io/files/ccr.pdf) tracker, whose Python implementation is underway (you can check the Matlab code [here](https://github.com/ESanchezLozano/iCCR)).

An example of usage is included in the first release. Full scripts for folder and csv reading will follow soon.

## Requirements
dlib --> pip install dlib [Link](https://pypi.org/project/dlib/)
OpenCV --> pip install cv2 [Link](http://opencv-python-tutroals.readthedocs.io/en/latest/)
PyTorch --> follow the steps in [https://pytorch.org/](https://pytorch.org/)

```
pip install dlib
pip install cv2
```

## Contributions

All contributions are welcome

## Citation

```
@inproceedings{sanchez2018bmvc,
  title = {Joint Action Unit localisation and intensity estimation through heatmap regression},
  author = {Enrique Sánchez-Lozano and Georgios Tzimiropoulos and Michel Valstar},
  booktitlte = {BMVC},
  year = 2018
}
```





