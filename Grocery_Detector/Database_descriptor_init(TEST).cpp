// Grocery_Detector.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\flann\flann.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "image_Details.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


void readImages();





int main(int argc, char** argv)
{

	String directory = "c:\\Users\\Wilhelm\\Desktop\\grocery_items\\*.jpg";
	vector <String> filenames;

	Mat query_image;
	query_image = imread("c:\\Users\\Wilhelm\\Desktop\\grocery_items\\IMG_0115.jpg");
	if (!query_image.data) {
		cout << "Failed to load query image." << endl;
		return -1;
	}

	glob(directory, filenames, false);
	for (int i = 0; i < filenames.size(); i++) {
		cout << "Image filename:::" << filenames[i] << endl;
	}
	cout << filenames.size() << endl;
	vector <Mat> image_database(filenames.size());

	for (int i = 0; i < filenames.size(); i++) {
		image_database[i] = imread(filenames[i]);
		if (!image_database[i].data) {
			cout << "Failed to open image" << endl;
		}
	}

	Ptr<SURF> extractor = SurfDescriptorExtractor::create(400);
	Ptr<SURF> detector = SurfFeatureDetector::create(400);
	Mat db_descriptors;
	Mat query_descriptor;
	vector <KeyPoint> queryKeypoints;

	detector->detect(query_image, queryKeypoints);
	extractor->compute(query_image, queryKeypoints, query_descriptor);

	vector < vector<KeyPoint> > keypoints(image_database.size(), vector <KeyPoint>());
	vector <Mat> descriptors(image_database.size());

	//Load all descriptors into vector.
	int size = filenames.size();
	image_Details* row_imgInfo = new image_Details[size];
	int rows = 0;
	for (int i = 0; i < size; i++) {
		cout << "Calculating descriptors for image::" << filenames[i] << endl;
		detector->detect(image_database[i], keypoints[i]);
		extractor->compute(image_database[i], keypoints[i], descriptors[i]);
		row_imgInfo[i].filename = filenames[i];
		row_imgInfo[i].rowStart = rows;
		row_imgInfo[i].rowEnd = rows + descriptors[i].rows - 1;
		rows = rows + descriptors[i].rows;
	}

	//concatenate all descriptors into a SUPER Matrix of database descriptors
	vconcat(descriptors, db_descriptors);



	ofstream rowsOut;
	ofstream imgInfoOut;

	imgInfoOut.open("c:\\Users\\Wilhelm\\Desktop\\imginfo.txt");
	imgInfoOut << filenames.size();
	for (int i = 0; i<size; i++) {
		imgInfoOut << row_imgInfo[i];
	}

	rowsOut.open("c:\\Users\\Wilhelm\\Desktop\\rowsOut.txt");
	for (int i = 0; i < size; i++) {
		rowsOut << row_imgInfo[i].rowStart << "::" << row_imgInfo[i].rowEnd << endl;
	}
	rowsOut.close();

	string fileName = "c:\\Users\\Wilhelm\\Desktop\\descriptor_DB.xml";
	FileStorage fs(fileName, FileStorage::WRITE);

	fs << "descriptor_DB" << db_descriptors;
	fs.release();
	getchar();
	waitKey();
	return 0;
}

