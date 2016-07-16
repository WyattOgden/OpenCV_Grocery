#include "stdafx.h"
#include "image_obj.h"
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

int main(int argc, char** argv) {


	Mat scene, final;
	vector<String> filenames;
	String directory = "c:\\Users\\Wilhelm\\Desktop\\image_database\\*.jpg";

	glob(directory, filenames, false);
	for (int i = 0; i < filenames.size(); i++) {
		cout << filenames[i] << endl;
	}
	vector<image_obj> images(filenames.size());
	for (int i = 0; i < filenames.size(); i++) {
		Mat src;
		src = imread(filenames[i]);
		images[i].image = src;
		if (!images[i].image.data) {
			cout << "Failed to open image" << endl;
		}

	}

	scene = imread("c:\\Users\\Wilhelm\\Desktop\\archive.jpg");
	if (!scene.data) {
		cout << "Failed to read in image for scene!" << endl;
		return -1;
	}

	final = scene;

	Ptr<SURF> detector = SURF::create(400);
	for (int i = 0; i < images.size(); i++) {
		detector->detectAndCompute(images[i].image, Mat(), images[i].keypoints, images[i].descriptors);
	}

	Mat scene_descriptors;
	vector<KeyPoint> scene_keypoints;
	detector->detectAndCompute(scene, Mat(), scene_keypoints, scene_descriptors);

	FlannBasedMatcher matcher;
	for (int i = 0; i < images.size(); i++) {
		matcher.match(images[i].descriptors, scene_descriptors, images[i].matches);
	}


	ofstream outfile;
	outfile.open("c:\\Users\\Wilhelm\\Desktop\\debug_test.txt");
	for (int i = 0; i < images.size(); i++) {
		outfile << "IMAGE OBJECT #" << i << endl;
		for (int j = 0; j < images[i].descriptors.rows; j++) {
			double dist = images[i].matches[j].distance;
			outfile << "DIST::: " << dist << endl;
			if (dist > images[i].max_dist) {
				images[i].max_dist = dist;
				cout << images[i].max_dist << ": MAX" << endl;
			}
			if (dist < images[i].min_dist) {
				images[i].min_dist = dist;
				cout << images[i].min_dist << ": MIN" << endl;
			}
		}
	}
	outfile.close();
	for (int i = 0; i < images.size(); i++) {
		images[i].findGoodMatches();
		cout << images[i].goodMatches.size() << endl;
	}

	//Mat img_matches;
	//drawMatches(images[1].image, images[1].keypoints, final, scene_keypoints,
	//	images[1].goodMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
	//	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//imshow("Matching test", img_matches);

	vector <Point2f> scene1;
	vector <Point2f> scene2;
	vector <Point2f> scene3;
	vector <Point2f> scene4;
	vector <Point2f> scene5;

	for (int i = 0; i < images[0].goodMatches.size(); i++) {
		images[0].obj.push_back(images[0].keypoints[images[0].goodMatches[i].queryIdx].pt);
		scene1.push_back(scene_keypoints[images[0].goodMatches[i].trainIdx].pt);
	}
	for (int i = 0; i < images[1].goodMatches.size(); i++) {
		images[1].obj.push_back(images[1].keypoints[images[1].goodMatches[i].queryIdx].pt);
		scene2.push_back(scene_keypoints[images[1].goodMatches[i].trainIdx].pt);
	}
	for (int i = 0; i < images[2].goodMatches.size(); i++) {
		images[2].obj.push_back(images[2].keypoints[images[2].goodMatches[i].queryIdx].pt);
		scene3.push_back(scene_keypoints[images[2].goodMatches[i].trainIdx].pt);
	}
	for (int i = 0; i < images[3].goodMatches.size(); i++) {
		images[3].obj.push_back(images[3].keypoints[images[3].goodMatches[i].queryIdx].pt);
		scene4.push_back(scene_keypoints[images[3].goodMatches[i].trainIdx].pt);
	}
	for (int i = 0; i < images[4].goodMatches.size(); i++) {
		images[4].obj.push_back(images[4].keypoints[images[4].goodMatches[i].queryIdx].pt);
		scene5.push_back(scene_keypoints[images[4].goodMatches[i].trainIdx].pt);
	}

	vector<Mat> H(images.size());

	H[0] = findHomography(images[0].obj, scene1, CV_RANSAC);
	H[1] = findHomography(images[1].obj, scene2, CV_RANSAC);
	H[2] = findHomography(images[2].obj, scene3, CV_RANSAC);
	H[3] = findHomography(images[3].obj, scene4, CV_RANSAC);
	H[4] = findHomography(images[4].obj, scene5, CV_RANSAC);

	vector< vector<Point2f> > scene_corners(images.size(), vector<Point2f>(4));

	for (int i = 0; i < images.size(); i++) {
		images[i].obj_corner.push_back(cvPoint(0, 0));
		images[i].obj_corner.push_back(cvPoint(images[i].image.cols, 0));
		images[i].obj_corner.push_back(cvPoint(images[i].image.cols, images[i].image.rows));
		images[i].obj_corner.push_back(cvPoint(0, images[i].image.rows));
	}
	for (int i = 0; i < images.size(); i++) {
		perspectiveTransform(images[i].obj_corner, scene_corners[i], H[i]);
	}


	for (int i = 0; i < images.size(); i++) {
		line(scene, scene_corners[i][0] , scene_corners[i][1] , Scalar(0, 255, 0), 4);
		line(scene, scene_corners[i][1] , scene_corners[i][2], Scalar(0, 255, 0), 4);
		line(scene, scene_corners[i][2] , scene_corners[i][3], Scalar(0, 255, 0), 4);
		line(scene, scene_corners[i][3] , scene_corners[i][0], Scalar(0, 255, 0), 4);
	}

	cv::imshow("Final Matching", scene);

	cv::waitKey(0);
	return 0;
}
