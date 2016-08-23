// ConsoleApplication3.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "image_Details.h"
#include "Found_Object.h"
#include "ROI.h"
#include <stdio.h>
#include <opencv2\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int window_rows = 256;
int window_cols = 341;
int rowstep = window_rows;
int colstep = window_cols;

void readImages(ifstream& inFile, int databaseSize, vector<String>& filenames, image_Details img_info[]) {
	for (int i = 0; i < databaseSize * 3; i++) {
		inFile >> img_info[i];
	}
	for (int i = 0; i < databaseSize; i++) {
		img_info[i].filename = filenames[i];
	}
	for (int i = 0; i < databaseSize; i++) {
		cout << img_info[i];
	}
	return;
}

void imageResetSim(image_Details obj[], int size) {
	for (int i = 0; i < size; i++) {
		obj[i].similarity = 0;
	}
}

void drawROIS(Mat query, vector<ROI>& Roi) {
	for (int row = 0; row <= query.rows - window_rows; row += rowstep) {
		for (int col = 0; col <= query.cols - window_cols; col += colstep) {

			Mat temp;
			Rect window(col, row, window_cols, window_rows);
			rectangle(query, window, Scalar(255, 0, 0), 1, 8, 0);
			temp = query(window);

			ROI roi(temp, window_cols, window_rows, col, row);
			Roi.push_back(roi);
		}
	}
}

void read_OfflineData(String directory, vector<String>& filenames, Mat& db_descriptors) {
	
	glob(directory, filenames, false);
	for (int i = 0; i < filenames.size(); i++) {
		cout << "Image filename:::" << filenames[i] << endl;
	}

	string filename = "c:\\Users\\Wilhelm\\Desktop\\Grocery Project\\descriptor_DB.xml";
	FileStorage fs(filename, FileStorage::READ);
	fs["descriptor_DB"] >> db_descriptors;
}

/** @function main */
int main(int argc, char** argv)
{
	//read offline data into database of descriptors
	String directory = "c:\\Users\\Wilhelm\\Desktop\\Grocery_List\\*.jpg";
	vector <String> filenames;
	int databaseSize;
	Mat db_descriptors;
	read_OfflineData(directory, filenames, db_descriptors);

	//read database image info in
	ifstream inFile;
	inFile.open("c:\\Users\\Wilhelm\\Desktop\\imginfo.txt");
	inFile >> databaseSize;
	image_Details* img_info = new image_Details[databaseSize];
	readImages(inFile, databaseSize, filenames, img_info);
	inFile.close();


	Mat query;
	string QueryString = "c:\\Users\\Wilhelm\\Desktop\\rsz_match_test.jpg";
	query = imread(QueryString);
	if (!query.data) {
		cout << "error loading image." << endl;
	}

	vector <ROI> Roi;
	cout << Roi.size() << endl;

	cout << query.rows << ":: " << query.cols << endl;
	drawROIS(query, Roi);
	cout << "TEEEEEESTTT" << endl;
	//INITIALIZE DETECTOR AND INDEXER(FOR SEARCH)
	Ptr<SURF> detector = SURF::create(400);
	vector<Found_Object> objectsFound;
	vector<KeyPoint> keypoints;
	Mat descriptors;

	detector->detectAndCompute(query, Mat(), keypoints, descriptors);
	//Find the keypoints that are contained within each region of interest and store them in the ROI object
	for (int i = 0; i < Roi.size(); i++) {
		Roi[i].calculateKeypoints(keypoints);
	}
	//print sizes for testing purposes
	for (int i = 0; i < Roi.size(); i++) {
		cout << Roi[i].keypoints.size() << endl;
	}
	//Search through all ROI's and find objects from them.
	flann::Index index(db_descriptors, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);
	cout << "FLann indexed" << endl;
	for (int c = 0; c < Roi.size(); c++) {
		//Compute descriptors for the keypoints within the ROI
		detector->compute(query, Roi[c].keypoints, Roi[c].descriptors);
		if (Roi[c].keypoints.size() <= 25) {
			continue;
		}
		//drawKeypoints(query, keypoints, query);
		Mat indices(Roi[c].descriptors.rows, 2, CV_32S);
		Mat distances(Roi[c].descriptors.rows, 2, CV_32F);


		index.knnSearch(Roi[c].descriptors, indices, distances, 2, 12);

		cout << "FLann indexed" << endl;
		//The the matched descriptor belongs to the image. SO INCREASE SIMILIARITY
		float* p;
		int* iP;
		bool similar = false;
		iP = indices.ptr<int>(0);
		p = distances.ptr<float>(0);
		for (int i = 0; i < indices.rows*indices.cols; i++) {
			if (p[i] < 0.8*p[i + 1]) {
				for (int j = 0; j < databaseSize; j++) {
					if (img_info[j].rowStart <= iP[i] && img_info[j].rowEnd >= iP[i + 1]) {
						img_info[j].similarity++;
						similar = true;
						break;
					}
				}
			}
			i++;
		}

		if (!similar) {
			continue;
		}
		cout << "Similarity matching !" << endl;
		int max = 0;
		int indexMax;

		for (int i = 0; i < databaseSize; i++) {
			cout << img_info[i].similarity << endl;
			if (max < img_info[i].similarity) {
				max = img_info[i].similarity;
				indexMax = i;
			}
		}

		if (img_info[indexMax].similarity >= 10) {
			cout << "Image matched! To ROI #" << c << "Filenam::" << filenames[indexMax] << endl;
			if (objectsFound.size() == 0) {
				Found_Object objectDetected(img_info[indexMax].filename, img_info[indexMax].rowStart, img_info[indexMax].rowEnd);
				objectDetected.matchedRegions.push_back(Roi[c]);
				objectsFound.push_back(objectDetected);
			}
			else {
				bool found = false;
				for (int i = 0; i < objectsFound.size(); i++) {
					if (filenames[indexMax] == objectsFound[i].objectName) {
						//IF object is found already, add the ROI to the found_objects ROI grouping
						objectsFound[i].matchedRegions.push_back(Roi[c]);
						found = true;
					}
				}
				if (!found) {	//If it wasnt found create a new object and add it to list
					Found_Object objectDetected(img_info[indexMax].filename, img_info[indexMax].rowStart, img_info[indexMax].rowEnd);
					objectDetected.matchedRegions.push_back(Roi[c]);
					objectsFound.push_back(objectDetected);
				}
			}
		}
		imageResetSim(img_info, databaseSize);
	}

	for (int i = 0; i < objectsFound.size(); i++) {
		cout << "OBJECT FOUND:" << endl;
		cout << objectsFound[i].objectName << endl;
	}

	for (int i = 0; i < objectsFound.size(); i++) {
		objectsFound[i].drawMatches_ROI(i);
		int check = objectsFound[i].compareGrouping(QueryString, index, img_info, databaseSize);
		if (check < 0) {
			continue;
		}
		else {
			cout << endl;
			cout << "Object that was originally found in seperate ROIS" << objectsFound[i].objectName << endl;
			cout << "After aggregation of ROIS and check:: " << filenames[check] << endl << endl << endl;
		}
	}
	imshow("Query", query);
	waitKey(0);
	getchar();
	return(0);
}

