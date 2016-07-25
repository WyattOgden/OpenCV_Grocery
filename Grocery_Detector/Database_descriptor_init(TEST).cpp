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

	Mat indices(query_descriptor.rows, 2, CV_32S);
	Mat distances(query_descriptor.rows, 2, CV_32F);

	flann::Index index(db_descriptors, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);
	index.knnSearch(query_descriptor, indices, distances, 2, 24);

	ofstream indicesOut;
	ofstream rowsOut;
	ofstream distanceOut;
	ofstream imgInfoOut;

	imgInfoOut.open("c:\\Users\\Wilhelm\\Desktop\\imginfo.txt");
	imgInfoOut << filenames.size();
	for (int i = 0; i<size; i++) {
		imgInfoOut << row_imgInfo[i];
	}

	indicesOut.open("c:\\Users\\Wilhelm\\Desktop\\IndicesOut.txt");
	indicesOut << indices;
	indicesOut.close();

	rowsOut.open("c:\\Users\\Wilhelm\\Desktop\\rowsOut.txt");
	for (int i = 0; i < size; i++) {
		rowsOut << row_imgInfo[i].rowStart << "::" << row_imgInfo[i].rowEnd << endl;
	}
	rowsOut.close();

	distanceOut.open("c:\\Users\\Wilhelm\\Desktop\\distanceOut.txt");
	distanceOut << distances;
	distanceOut.close();

	for (int i = 0; i < indices.rows; i++) {
		if (distances.at<float>(i, 0) < (0.6*distances.at<float>(i, 1))) {
			for (int j = 0; j < size; j++) {
				if (row_imgInfo[j].rowStart <= indices.at<int>(i, 0) && row_imgInfo[j].rowEnd >= indices.at<int>(i, 1)) {
					row_imgInfo[j].similarity++;
					break;
				}
			}
		}
	}

	int max = row_imgInfo[0].similarity;
	int indexMax = 0;
	for (int i = 0; i < size; i++) {
		cout << row_imgInfo[i].similarity << endl;
	}
	for (int i = 1; i < size; i++) {
		if (max < row_imgInfo[i].similarity) {
			max = row_imgInfo[i].similarity;
			indexMax = i;
		}
	}
	cout << "Image matched!" << filenames[indexMax] << endl;
	string fileName = "c:\\Users\\Wilhelm\\Desktop\\descriptor_DB.xml";
	FileStorage fs(fileName, FileStorage::WRITE);

	fs << "descriptor_DB" << db_descriptors;
	fs.release();
	getchar();
	waitKey();
	return 0;
}

