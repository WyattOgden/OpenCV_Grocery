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
#include <json\json.h>
#include <json\writer.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//Reads image info from the initTest cpp file that calculates the database of descriptors. And exports rowStart and rowEnd info
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

}

//Function that resets similarity after each iteration of the loop
void imageResetSim(image_Details obj[], int size) {
	for (int i = 0; i < size; i++) {
		obj[i].similarity = 0;
	}
}

void parsePriceName(string filename, string& price, string& name, int startIndex) {

	char temp;
	int i = startIndex;
	//read in name until '_'
	for (i; filename[i] != '_'; i++) {
		temp = filename[i];
		name += temp;
	}
	//read in price until '_'
	i += 1;
	for (i; filename[i] != '_'; i++) {
		//if (filename[i] == '$') {
		//	temp = '.';
		//	price += temp;
		//	continue;
		//}

		temp = filename[i];
		price += temp;
	}

}


int main(int argc, char** argv)
{
	//Read in the names of all the images within the directory
	String directory = "c:\\Users\\Wilhelm\\Desktop\\grocery_items\\*.jpg";
	string shit = "c:\\Users\\Wilhelm\\Desktop\\grocery_items\\";
	int startingIndex = shit.size();
	vector <String> filenames;

	glob(directory, filenames, false);
	for (int i = 0; i < filenames.size(); i++) {
		cout << "Image filename:::" << filenames[i] << endl;
	}
	cout << filenames.size() << endl;
	//Creates VideoCapture Object
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		return -1;
	}

	//Reads the XML file that contains the SUPER MATRIX of descriptors for ALL images in the database.
	//Places the descriptors into db_descriptors Matrix, use this matrix to search the index and match to query.
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
	//Uses function to read in all important info into the array of image_Details objects, each object represents 1 images info
	readImages(inFile, databaseSize, filenames, img_info);

	//Creates a SURF object used to calculate keypoints and descriptors from the query image.
	Ptr<SURF> detector = SURF::create(400);
	//Indexes all of the descriptors from the db_descriptors SUPER DATABASE for better searching
	flann::Index index(db_descriptors, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);
	//LOOP TO DETECT ITEM FROM CAMERA FEED
	int frameCount = 0;

	string prevItem = " ";
	ofstream outFile;
	outFile.open("c:\\Users\\Wilhelm\\Desktop\\outJSON.txt");
	if (!outFile) {
		cout << "Shit failed dawg." << endl;
		return -1;
	}

	while (1) {

		Mat frame;
		//Capture the current fram into a Mat, basically treating it as an image.
		capture >> frame;

		//calculate features.
		if (frameCount % 2 == 0) {
			//Keypoints and descriptors for the query image to be searched and matched
			vector<KeyPoint> keypoints;
			Mat descriptors;

			//find keypoints and descriptors
			detector->detectAndCompute(frame, Mat(), keypoints, descriptors);
			//These matrices are passed to the flann.knnsearch as parameters.
			Mat indices(descriptors.rows, 2, CV_32S);
			Mat distances(descriptors.rows, 2, CV_32F);

			//Draws all the keypoints onto the frame  (for testing purposes to see how they are clustered)
			drawKeypoints(frame, keypoints, frame, DrawMatchesFlags::DEFAULT);
			imshow("Keypoints", frame);
			//Ticks are used to calculate the time it takes for algorithims to finish

			//the previous index we created from SUPER database of descriptors(Imported from offline) calls this function.
			// Takes the descriptors of the query image as argument. Searches index and stores the matches in distances
			// The indices of the distances within the Databse of descriptors are stored in indices.
			index.knnSearch(descriptors, indices, distances, 2, 24);

			//This for loop cycles through the entire distance/indices matrices 
			//Compares the distance in the first column with the 2nd(multiplied by a constant).
			//When a good distance is found enter 2nd for loop
			//The 2nd for loop goes through each images info (stored in img_info object array) and attempts to match
			// The distances (from distance matrix) index (index of corresponding distance stored in indices) with the image object
			//Whose descriptors fill some interval. If this index of the good distance match fits within this interval (rowStart -> rowEnd)
			//Then the matched descriptor belongs to the image. SO INCREASE SIMILIARITY
			float* p;
			int* iP;
			iP = indices.ptr<int>(0);
			p = distances.ptr<float>(0);
	
			for (int i = 0; i < indices.rows*indices.cols; i++) {
				if (p[i] < 0.8*p[i+1]) {
					for (int j = 0; j < databaseSize; j++) {
						if (img_info[j].rowStart <= iP[i] && img_info[j].rowEnd >= iP[i+1]) {
							img_info[j].similarity++;
							break;
						}
					}
				}
				i++;
			}


			//Object with the most similarity is the object that has been found!
			int max = 0;
			int indexMax;
		
			for (int i = 0; i < databaseSize; i++) {
				if (max < img_info[i].similarity) {
					max = img_info[i].similarity;
					indexMax = i;
				}
			}

			if (img_info[indexMax].similarity > 40) {
				cout << "Image matched!" << filenames[indexMax] << endl;
				//If looking at the same object, do nothing
				string price;
				string name;
				parsePriceName(filenames[indexMax], price, name, startingIndex);
				if (prevItem == name) {
					imageResetSim(img_info, databaseSize);
					continue;
				}
				//If different found object, return i
				else {
					Json::Value JSONevent;
					JSONevent[name]["price"] = price;
					string filenameOut = filenames[indexMax];
					JSONevent[name]["filename"] = filenameOut;
					cout << JSONevent << endl;
					outFile << JSONevent << endl;
				}
				prevItem = name;
			}
			//Reset all of the objects similarity counts for the next iteration
			imageResetSim(img_info, databaseSize);
			if (waitKey(30) >= 0) {
				break;
			}
		}

		if (waitKey(30) >= 0) {
			break;
		}
		frameCount++;
	}

	return 0;
}

