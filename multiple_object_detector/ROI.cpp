#include "stdafx.h"
#include "ROI.h"
#include "image_Details.h"


ROI::ROI()
{
}

ROI::ROI(Mat& imageIn, int colWidth, int rowHeight, int X, int Y) {
	image = imageIn.clone();

	width = colWidth;
	height = rowHeight;

	origX = X;
	origY = Y;
}

void ROI::calculateKeypoints(vector<KeyPoint> origKey) {
	int xBound = width + origX;
	int yBound = height + origY;

	for (int i = 0; i < origKey.size(); i++) {
		if (origKey[i].pt.x <= xBound && origKey[i].pt.x >= origX && origKey[i].pt.y <= yBound && origKey[i].pt.y >= origY) {
			keypoints.push_back(origKey[i]);
		}
	}
}

ROI::~ROI()
{
}
