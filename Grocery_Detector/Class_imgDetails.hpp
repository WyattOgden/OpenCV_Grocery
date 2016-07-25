#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

class image_Details{
public:
	image_Details();

	String filename;
	int rowStart;
	int rowEnd;
	int similarity = 0;

	ofstream& operator << (ofstream& outfile, image_Details& img_info);
	void operator >> (ifstream& infile, image_Details& img_info);
};


ofstream& image_Details:: operator << (ofstream& outfile, image_Details& img_info){
	outfile << filename << endl;
	outfile << rowStart << endl;
	outfile << rowEnd << endl;
	outfile << similarity << endl;
}

void image_Details:: operator >>(ifstream& infile, image_Details& img_info){
	infile >> filename;
	infile >> rowStart;
	infile >> rowEnd;
	infile >> similarity;
}
