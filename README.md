# Udacity-SDCND---P1-Lane-Finding
My first project for the Udacity Self-driving Car Nanodegree, finding lane lines using Canny edge detection and the Hough transform.

# **Finding Lane Lines on the Road** 
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

## Test on Images

```python
import os
os.listdir("test_images/")
```




    ['solidYellowLeft.jpg',
     'solidWhiteCurve.jpg',
     'solidWhiteRight.jpg',
     'solidYellowCurve2.jpg',
     'whiteCarLaneSwitch.jpg',
     'solidYellowCurve.jpg']




```python
def apply_yellow_white_color_mask(image):
    # White Colour Mask
    lower_white = np.array([200, 200, 200], dtype="uint8")*np.sqrt(np.mean(image)/30)
    upper_white = np.array([255, 255, 255], dtype="uint8")
    mask_white = cv2.inRange(image, np.array(lower_white.astype(int)), upper_white.astype(int))
        
    # Yellow Colour Mask
    lower_yellow = np.array([175, 175, 0], dtype="uint8")
    upper_yellow = np.array([255, 255, 175], dtype="uint8")
    mask_yellow_RGB = cv2.inRange(image, lower_yellow.astype(int), upper_yellow.astype(int))
        
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lower_yellow_HLS = np.array([95, 150, 90], dtype="uint8")
    upper_yellow_HLS = np.array([140, 200, 255], dtype="uint8")
    mask_yellow_HLS = cv2.inRange(img, lower_yellow_HLS.astype(int), upper_yellow_HLS.astype(int))
        
    mask_yellow = cv2.bitwise_and(mask_yellow_HLS, mask_yellow_RGB)
        
    #Add White & Yellow Colour Mask
    mask = cv2.add(mask_yellow, mask_white)
    return cv2.bitwise_and(image, image, mask = mask)
```


```python
def mask_image_region(image):
    im_height = image.shape[0]
    im_width = image.shape[1]
    
    vertices=np.array([[(im_width*0.2/10, im_height),(im_width/2.1, im_height/1.8), (im_width/1.9, im_height/1.8), (im_width*9.8/10, im_height)]])
    return region_of_interest(img=image, vertices = np.int32(vertices))
```


```python
num_images = len(os.listdir("test_images/"))
plt.figure(figsize=(20, num_images*3))

index = 0
for image_dir in os.listdir("test_images/"):
    image = mpimg.imread('test_images/'+image_dir)
    reset_line_values()
    
    # REGION MASK
    region_masked_image = mask_image_region(image)
    
    # YELLOW & WHITE COLOUR MASK
    colour_masked_image = apply_yellow_white_color_mask(region_masked_image)
    
    # GRAYSCALE
    gray_image = grayscale(colour_masked_image)
    
    # BLUR
    kernel_size = 5
    blurred_gray_image = gaussian_blur(gray_image, kernel_size)

    # EDGE DETECTION
    low_threshold = 80
    high_threshold = 270
    image_edges = cv2.Canny(blurred_gray_image, low_threshold, high_threshold)
    
    # LINE DETECTION
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_length = 60
    max_line_gap = 40
    image_lines = hough_lines(img=image_edges, max_line_gap=max_line_gap, min_line_len=min_line_length, rho=rho, theta=theta, threshold=threshold)
    
    # RESULT
    result = weighted_img(image, image_lines)
    
    # PLOTS
    plt.subplot(num_images, 4, index+1)
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(num_images, 4, index+2)
    plt.imshow(colour_masked_image)
    plt.axis('off')
    
    plt.subplot(num_images, 4, index+3)
    plt.imshow(image_edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(num_images, 4, index+4)
    plt.imshow(result)
    plt.axis('off')
    
    # SAVE IMAGE
    plt.imsave("./test_images/RESULT_"+image_dir, result)
    
    index = index + 4
    
plt.show()
```

    /home/gerhard/anaconda3/lib/python3.5/site-packages/ipykernel/__main__.py:47: FutureWarning: comparison to `None` will result in an elementwise object comparison in the future.
    /home/gerhard/anaconda3/lib/python3.5/site-packages/scipy/stats/_stats_mstats_common.py:97: RuntimeWarning: invalid value encountered in double_scalars
      sterrest = np.sqrt((1 - r**2) * ssym / ssxm / df)



![png](output_14_1.png)

## Test on Videos



Let's try the one with the solid white lane on the right first ...


<video width="960" height="540" controls>
  <source src="white.mp4">
</video>




**At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

Now for the one with the solid yellow lane on the left. This one's more tricky!


<video width="960" height="540" controls>
  <source src="yellow.mp4">
</video>


## Optional Challenge


[![](https://img.youtube.com/vi/Wb_AaCSJJAo/0.jpg)](https://youtu.be/Wb_AaCSJJAo)


<video width="960" height="540" controls>
  <source src="extra.mp4">
</video>



