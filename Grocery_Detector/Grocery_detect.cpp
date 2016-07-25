// Grocery_Detector.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "image_Details.h"
#include <stdio.h>
#include <opencv2\imgproc.hpp>
#include <opencv2\flann\flann.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void readImages();

void imageResetSim(image_Details obj[], int size) {
	for (int i = 0; i < size; i++) {
		obj[i].similarity = 0;
	}
}



int main(int argc, char** argv)
{

	String directory = "c:\\Users\\Wilhelm\\Desktop\\grocery_items\\*.jpg";
	vector <String> filenames;

	glob(directory, filenames, false);
	for (int i = 0; i < filenames.size(); i++) {
		cout << "Image filename:::" << filenames[i] << endl;
	}
	cout << filenames.size() << endl;

	VideoCapture capture(0);
	if (!capture.isOpened()) {
		return -1;
	}
	string filename = "c:\\Users\\Wilhelm\\Desktop\\Grocery Project\\descriptor_DB.xml";
	FileStorage fs(filename, FileStorage::READ);
	Mat db_descriptors;
	fs["descriptor_DB"] >> db_descriptors;

	//Read in Image Info from offline 
	int databaseSize;
	ifstream inFile;
	inFile.open("c:\\Users\\Wilhelm\\Desktop\\imginfo.txt");
	inFile >> databaseSize;
	image_Details* img_info = new image_Details[databaseSize];
	for (int i = 0; i < databaseSize * 3; i++) {
		inFile >> img_info[i];
	}
	for (int i = 0; i < databaseSize; i++) {
		img_info[i].filename = filenames[i];
	}
	for (int i = 0; i < databaseSize; i++) {
		cout << img_info[i];
	}

	Ptr<SURF> detector = SURF::create(400);
	
	//LOOP TO DETECT ITEM FROM CAMERA FEED
	int frameCount = 0;
	while (1) {

		Mat frame;
		capture >> frame;
		imshow("Vid capture", frame);

		//calculate features.
		if (frameCount % 2 == 0) {
			vector<KeyPoint> keypoints;
			Mat descriptors;

			detector->detectAndCompute(frame, Mat(), keypoints, descriptors);
			Mat indices(descriptors.rows, 2, CV_32S);
			Mat distances(descriptors.rows, 2, CV_32F);

			flann::Index index(db_descriptors, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);
			index.knnSearch(descriptors, indices, distances, 2, 24);

			for (int i = 0; i < indices.rows; i++) {
				if (distances.at<float>(i, 0) < (0.8*distances.at<float>(i, 1))) {
					for (int j = 0; j < databaseSize; j++) {
						if (img_info[j].rowStart <= indices.at<int>(i, 0) && img_info[j].rowEnd >= indices.at<int>(i, 1)) {
							img_info[j].similarity++;
							break;
						}
					}
				}
			}

			int max = img_info[0].similarity;
			int indexMax = 0;
			
			for (int i = 1; i < databaseSize; i++) {
				if (max < img_info[i].similarity) {
					max = img_info[i].similarity;
					indexMax = i;
				}
			}
			if (img_info[indexMax].similarity > 25) {
				cout << "Image matched!" << filenames[indexMax] << endl;
			}
			imageResetSim(img_info, databaseSize);
		}

		if (waitKey(30) >= 0) {
			break;
		}
		frameCount++;
	}

	return 0;
}

