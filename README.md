# Udacity SDCND - P1 Lane-Finding
My first project for the Udacity Self-driving Car Nanodegree, finding lane lines using Canny edge detection and the Hough transform.

# **Finding Lane Lines on the Road** 
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

## Test on Images

![png](output_14_1.png)

## Test on Videos

Let's try the one with the solid white lane on the right first ...


https://youtu.be/M9eXh16gSxM

[![](https://img.youtube.com/vi/M9eXh16gSxM/0.jpg)](https://youtu.be/M9eXh16gSxM)


**At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

Now for the one with the solid yellow lane on the left. This one's more tricky!

https://youtu.be/Q4gycJoBKdo

[![](https://img.youtube.com/vi/Q4gycJoBKdo/0.jpg)](https://youtu.be/Q4gycJoBKdo)


## Optional Challenge

https://youtu.be/Wb_AaCSJJAo

[![](https://img.youtube.com/vi/Wb_AaCSJJAo/0.jpg)](https://youtu.be/Wb_AaCSJJAo)




