# HyperCam Data Processing Pipeline

**[`FOCUS Lab`](https://focus.ece.ufl.edu) × `SiTS Project`**

Developed and maintained by *[Yuxuan Zhang](mailto:zhangyuxuan@ufl.edu)*

---

## Stages of the processing pipeline

1. **Calibration**

	To be edited

1. **Alignment**

	To be edited

## Structure of `data` folder

```
.
└── data
    │
    ├── CAL_CHECKER -> Checkerboard images for distortion calibration
    │   ├── info.ini # Checker board parameters
    │   │              see template/check.info.ini
    │   ├── 1.png
    │   ├── 2.png
    │   └── ...
    │
    ├── CAL_WHITE -> White images here for white field calibration
    │   ├── Blue.png
    │   ├── Green.png
    │   ├── Infrared.png
    │   ├── Orange.png
    │   ├── Red.png
    │   ├── Ultra_Violet.png
    │   ├── Yellow_Green.png
    │   └── Yellow.png
    │
    ├── RAW -> Raw images captured by the low cost camera
    │   ├── XX_Blue.png
    │   ├── XX_Green.png
    │   ├── XX_Infrared.png
    │   ├── XX_Orange.png
    │   ├── XX_Red.png
    │   ├── XX_Ultra_Violet.png
    │   ├── XX_Yellow_Green.png
    │   ├── XX_Yellow.png
    │   └── ...
    │
    ├── REF -> Reference images captured by the high quality camera
    │   ├── info.ini # Reference image parameters
    │   │              see template/reference.info.ini
    │   ├── XX.dat
    │   ├── XX.hdr
    │   └── ...
    │
    ├── 1-Calibrated
    │   └── [Runtime generated images]
    │
    ├── 2-Aligned
    │   └── [Runtime generated images]
    │
    ├── 3-GridView
    │   └── [Runtime generated images]
    │
    └── report.txt
```