#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

class image_Details {
public:
	image_Details();
	~image_Details();
	string filename;
	int rowStart;
	int rowEnd;
	int similarity = 0;

	friend ostream& operator << (ostream& outfile, image_Details& img_info);
	friend istream& operator >> (ifstream& infile, image_Details& img_info);
};

