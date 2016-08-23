#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>
#include "ROI.h"
#include "image_Details.h"
#include <fstream>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
class Found_Object
{
public:
	Found_Object();
	Found_Object(string filename, int rowStart, int rowEnd);

	Ptr<SURF> detector = SURF::create(400);

	//Corresponding database object properties
	string objectName;
	Mat descriptors;
	vector<KeyPoint> keypoints;

	vector<ROI> matchedRegions;
	vector<DMatch> good_matches;
	Mat matchedImage;

	int xOrig;
	int yOrig;

	int width;
	int height;

	void calculate_Database_Keypoints();
	Mat combine_DrawROIS(string filename);
	void drawMatches_ROI(int count);
	//SPLIT VERIFY FUNCTION INTO SEVERAL FUNCTIONS THAT ARE ALL CALLED WITHIN
	int compareGrouping(string filename, flann::Index& index, image_Details row_imginfo[], int databaseSize);

	~Found_Object();
};

