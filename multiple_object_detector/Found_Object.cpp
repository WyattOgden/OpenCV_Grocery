#include "stdafx.h"
#include "Found_Object.h"
#include <opencv2\flann\flann.hpp>

Found_Object::Found_Object()
{
}

Found_Object::Found_Object(string filename1, int rowStart, int rowEnd) {
	//copy descriptors of this item type into the objects local descriptor Mat
	objectName = filename1;

	calculate_Database_Keypoints();
}

//THIS FUNCTION SHOULD BE REPLACED BY AN OFFLINE FUNCTION THAT EXPORTS THE DATA
void Found_Object::calculate_Database_Keypoints() {
	Mat image;
	image = imread(objectName);

	detector->detectAndCompute(image,Mat(), keypoints,descriptors);
}

//Draws all of the ROI's that correspond to the found object
Mat Found_Object::combine_DrawROIS(string filename) {

		//If there is more than 1 ROI for this found object, aggregate them into one single larger ROI and match again to Index, check if same result.
		int lowestX = matchedRegions[0].origX;
		int highestX = matchedRegions[0].origX + matchedRegions[0].width;

		int lowestY = matchedRegions[0].origY;
		int highestY = matchedRegions[0].origY + matchedRegions[0].height;

		int newWidth;
		int newHeight;
		//Find the bounds of the new ROI region
		for (int i = 1; i < matchedRegions.size(); i++) {
			if (matchedRegions[i].origX < lowestX) {
				lowestX = matchedRegions[i].origX;
			}
			if (matchedRegions[i].origX >= highestX) {
				highestX = matchedRegions[i].origX + matchedRegions[i].width;
			}
			if (matchedRegions[i].origY < lowestY) {
				lowestY = matchedRegions[i].origY;
			}
			if (matchedRegions[i].origY >= highestY) {
				highestY = matchedRegions[i].origY + matchedRegions[i].height;
			}
		}
		newWidth = highestX - lowestX;
		newHeight = highestY - lowestY;
		//DRAW THE ROI REGION IN THE IMAGE
		Mat outputImage = imread(filename);
		Rect window(lowestX, lowestY, newWidth, newHeight);
		rectangle(outputImage, window, Scalar(255, 0, 0), 1, 8, 0);
		imshow(objectName, outputImage);

		return(outputImage(window));
}

//Draws matches between ROI and the database image
void Found_Object::drawMatches_ROI(int count){
	vector<Mat> outputImages(matchedRegions.size());
	vector<Mat> localDescriptors(matchedRegions.size());
	Mat databaseImage = imread(objectName);

	char windowName[12];
	FlannBasedMatcher matcher;
	for (int i = 0; i < matchedRegions.size(); i++) {

		vector<DMatch> ROImatches;
		vector<DMatch> goodMatches;
		detector->detectAndCompute(matchedRegions[i].image, Mat(), matchedRegions[i].keypoints, localDescriptors[i]);
		
		matcher.match(localDescriptors[i], descriptors, ROImatches);
		//Find the min and max of the distances
		double max_dist = 0; double min_dist = 100;
		for (int i = 0; i < ROImatches.size(); i++)
		{
			double dist = ROImatches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		//Get only good matches
		for (int i = 0; i < ROImatches.size(); i++)
		{
			if (ROImatches[i].distance < 3*min_dist)
			{
				goodMatches.push_back(ROImatches[i]);
			}
		}
		drawMatches(matchedRegions[i].image, matchedRegions[i].keypoints, databaseImage, keypoints
			, goodMatches, outputImages[i], Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		sprintf(windowName, "%dWindow #%d", count, i);
		imshow(windowName, outputImages[i]);
		matchedImage = outputImages[i];
	}
}

//FUNCTION NEEDS TO BE RENAMED, WILL BE APART OF VERIFY FUNCTION LATER
int Found_Object::compareGrouping(string filename, flann::Index& index, image_Details img_info[], int databaseSize) {
	//FIRST CHECK!
	if (matchedRegions.size() > 1) {
		//If there is more than 1 ROI for this found object, aggregate them into one single larger ROI and match again to Index, check if same result.
		//THIS BELONGS IN COMBINE_DRAWROI function
		Mat grouping = combine_DrawROIS(filename);
		//DETECT AND COMPUTE KEYPOINT DESCRIPTORS
		vector<KeyPoint> gr_keypoints;
		Mat gr_descriptors;
		detector->detectAndCompute(grouping, Mat(), gr_keypoints, gr_descriptors);
		//MATCH TO DATABASE
		Mat indices(gr_descriptors.rows, 2, CV_32S);
		Mat distances(gr_descriptors.rows, 2, CV_32F);

		index.knnSearch(gr_descriptors, indices, distances, 2, 24);
		//Find most similar object!
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
			return -1;
		}
		//FIND THE INDEX OF THE MOST SIMILAR OBJECT
		int max = 0;
		int indexMax;
		for (int i = 0; i < databaseSize; i++) {
			cout << img_info[i].similarity << endl;
			if (max < img_info[i].similarity) {
				max = img_info[i].similarity;
				indexMax = i;
			}
		}
		FlannBasedMatcher matcher;
		

		vector<DMatch> ROImatches;
		vector<DMatch> goodMatches;


		matcher.match(gr_descriptors, descriptors, ROImatches);
		//Find the min and max of the distances
		double max_dist = 0; double min_dist = 100;
		for (int i = 0; i < ROImatches.size(); i++)
		{
			double dist = ROImatches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		//Get only good matches
		for (int i = 0; i < ROImatches.size(); i++)
		{
			if (ROImatches[i].distance < .8*min_dist)
			{
				goodMatches.push_back(ROImatches[i]);
			}
		}
		Mat outImage = imread(objectName);
		Mat out;
		drawMatches(grouping, gr_keypoints, outImage, keypoints
			, goodMatches, out, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		char shit[20];
		int random = rand();
		sprintf(shit, "Window %d",random);
		imshow(shit, out);
		return indexMax;
	}
	else {
		return -1;
	}
}

Found_Object::~Found_Object()
{
}
