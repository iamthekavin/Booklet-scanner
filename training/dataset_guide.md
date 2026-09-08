# VEE Scanner - Booklet Detection Annotation Guide

This guide covers the process for collecting and annotating data to train the YOLO11 booklet detection model.

## 1. Data Collection

**Minimum required:** 300-500 images

Capture images representing the actual scanning conditions:
- **Angles:** Top-down, slightly tilted (up to 20-30 degrees)
- **Lighting:** Natural room lighting, bright overhead lights, dim lighting, shadows falling across the desk/booklet.
- **Backgrounds:** Different desk textures (wood, white, dark, textured).
- **Variations:** 
  - Open booklet pages, closed front/back covers.
  - Hands holding or turning pages.
  - Background clutter (pens, ID cards, other papers).

## 2. Annotation Format (YOLO)

YOLO expects a `.txt` file for every `.jpg` file, containing one line per bounding box:
`<class_id> <x_center> <y_center> <width> <height>`

Coordinates are normalized between 0.0 and 1.0 based on image dimensions.

**Classes:**
* `0`: booklet
* `1`: hand
* `2`: background_clutter

## 3. Directory Structure

Your dataset should look like this before training:
```
dataset/
├── images/
│   ├── train/  (80% of data)
│   ├── val/    (20% of data)
│   └── test/   (optional)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## 4. Annotation Tools

### Option A: Roboflow (Recommended for Teams)
1. Create a Roboflow account and a new Object Detection project.
2. Upload your raw images.
3. Use the web-based bounding box tool:
   - Draw tight boxes around the `booklet`.
   - Box any visible `hand`s over the booklet or desk.
   - Box prominent `background_clutter`.
4. Click "Generate" and apply standard resizing (e.g., 640x640).
5. Export format: **YOLO v8/v11 (PyTorch)**. Download and extract directly to `d:\VEE_SCAN\training\dataset`.

### Option B: CVAT (Local & Open Source)
1. Install and run CVAT via Docker.
2. Create a new task and define labels: `booklet`, `hand`, `background_clutter`.
3. Upload images and annotate using the bounding box tool.
4. Export task dataset as **YOLO 1.1**.
5. Reorganize the exported folder structure into the `images` and `labels` splits shown above.

## 5. Tips for Quality Annotations
- **Consistency is key.** If you decide to include the spiral binding in the "booklet" box, do it in every image.
- **Occlusions:** If a hand covers part of the booklet, the booklet bounding box should still encompass the *entire inferred shape* of the booklet (or tightly bound the visible parts, depending on your chosen convention—just be consistent).
- **Tightness:** Draw boxes as tightly as possible around the object without cutting off edges.
