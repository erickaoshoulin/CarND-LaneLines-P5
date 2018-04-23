

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

[image1_0]: ./ipynb_images/car_vs_not_car.png
[image1_1]: ./ipynb_images/car_vs_not_car_hog.png
[image1_2]: ./ipynb_images/car_vs_not_car_bin_spatial_0.png
[image1_3]: ./ipynb_images/car_vs_not_car_bin_spatial_1.png
[image1_4]: ./ipynb_images/car_vs_not_car_bin_spatial_2.png


[image3_0]: ./ipynb_images/sliding_search_windows_0.png
[image3_1]: ./ipynb_images/sliding_search_windows_1.png
[image4_0]: ./ipynb_images/sliding_search_windows_2.png

[image5_0]:  ./test_failed_images/false_0.png
[image5_1]:  ./test_failed_images/false_1.png
[image5_2]:  ./test_failed_images/false_2.png
[image5_3]:  ./test_failed_images/false_3.png






## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of `vehicle_detection.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1_0]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image1_1]
![alt text][image1_2]
![alt text][image1_3]
![alt text][image1_4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3_0]

![alt text][image4_0]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. 
```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```
 Here are some example images:

![alt text][image3_1]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


I try to eyeball check the result by applying different parameters. Please see `Hog Sub-sampling Window Search` in `vehicle_detection.ipynb`.

```python
scales = (1, 1.55, 2.1)
ystarts = (380, 420, 420)
ystops = (500,  500, 550)
```


I try to dump the false positive in `project_video_out.mp4`, then retune the `find_car()` function. Here's line to false positive samples [link to test failed images](./test_failed_images)


![alt text][image5_0]
![alt text][image5_1]


### Here are six frames and their corresponding heatmaps:



### Here the resulting bounding boxes are drawn onto the last frame in the series:
Heat maps are accumulated through `num_frame_avg_m1` frames to avarage the noise. 

I try to acculuate `num_frame_avg_m1 = 28` frames heat map and set heat_trehold to 12 to filter out some false positive.
Please see `Tracking pipeline` in `vehicle_detection.ipynb`


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
false positive happen on yellow line. It's hard to get rid of it. I try to accumulate multiple frame and multiple scale to average the noise, but it's still there. I can't get rid of the noise completely.  
I also add some threshold for `draw_labeled_bboxes` to avoid draw some small box.

```python
  if((bbox[1][0]-bbox[0][0]) > 6) :
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
```

Here's final result [link to my video result](./project_video_out.mp4)

