#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;
class ROI
{
public:
	ROI();
	ROI(Mat& image, int width, int heigth, int x, int y);
	Mat image;
	vector<KeyPoint> keypoints;
	Mat descriptors;

	//Dimensions of the Region of Interest
	int width;
	int height;

	//Location of the ROI within the original image
	int origX;
	int origY;
	//Calculates the Keypoints within the ROI from the original image
	void calculateKeypoints(vector<KeyPoint> origKey);
	~ROI();
};

