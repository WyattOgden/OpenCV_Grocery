#include "stdafx.h"
#include "image_Details.h"


image_Details::image_Details()
{
}

ostream& operator << (ostream& outfile, image_Details& img_info) {
	outfile << img_info.rowStart << endl;
	outfile << img_info.rowEnd << endl;
	outfile << img_info.similarity << endl;
	return outfile;
}

istream& operator >>(ifstream& infile, image_Details& img_info) {
	infile >> img_info.rowStart;
	infile >> img_info.rowEnd;
	infile >> img_info.similarity;
	return infile;
}

image_Details::~image_Details()
{
}
