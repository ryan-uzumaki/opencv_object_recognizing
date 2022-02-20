#pragma once
#include <algorithm>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "opencv2/objdetect.hpp"
#include "stdlib.h"
#include "core/core.hpp"  
#include "highgui/highgui.hpp"  
#include "imgproc/imgproc.hpp"  
#include <cmath>
#include <sstream>
#include <iterator>

using namespace std;
using namespace cv;
//
//template<class ForwardIterator>
//inline size_t argmin(ForwardIterator first, ForwardIterator last)
//{
//	return std::distance(first, std::min_element(first, last));
//}
//
//template<class ForwardIterator>
//inline size_t argmax(ForwardIterator first, ForwardIterator last)
//{
//	return std::distance(first, std::max_element(first, last));
//}

class Process {
public:
	double get_distance(int W, int P);
	//Point frame_1_center_point(Mat& image);
	string Convert(float Num);
	void object_recognition(Mat& image);
};

template<class ForwardIterator>
inline size_t argmin(ForwardIterator first, ForwardIterator last)
{
	return std::distance(first, std::min_element(first, last));
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
	return std::distance(first, std::max_element(first, last));
}

string Process::Convert(float Num)
{
	std::ostringstream oss;
	oss << Num;
	std::string str(oss.str());
	return str;
}


double Process::get_distance(int W, int P) {
	double F = 550;
	double D = 0;
	D = (W * F) / P;
	return D;
}


void Process::object_recognition(Mat& image) {
	Process pr;
	int known_W = 9;
	int known_P = 510;
	Mat temp = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	addWeighted(image, 0.19, m, 0.0, 0, temp);
	Mat dst;
	bilateralFilter(temp, dst, 5, 20, 20);
	Mat m_ResImg;
	cvtColor(dst, m_ResImg, COLOR_BGR2HSV);
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	erode(m_ResImg, m_ResImg, element);//进行腐蚀操作
	erode(m_ResImg, m_ResImg, element);//进行腐蚀操作
	erode(m_ResImg, m_ResImg, element);//进行腐蚀操作
	Mat mask;
	inRange(m_ResImg, Scalar(100, 43, 46), Scalar(124, 255, 255), mask);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	vector<double> area;
	for (int i = 0; i < contours.size(); i++) {
		area.push_back(contourArea(contours[i]));
	}
	size_t maxIndex = distance(area.begin(), max_element(area.begin(), area.end()));
	Rect ret_1 = boundingRect(contours[maxIndex]);
	int avgX, avgY;
	avgX = (ret_1.x + ret_1.width) / 2;//x-axis middle point
	avgY = (ret_1.y + ret_1.height) / 2;//y-axis middle point
	for (int i = 0; i < contours.size(); i++) {
		//for (int j = 0; j < contours[i].size(); j++) {
		//	Point P = Point(contours[i][j].x, contours[i][j].y);
		//	Mat Contours = Mat::zeros(m_ResImg.size(), CV_8UC1);  //绘制
		//	Contours.at<uchar>(P) = 255;
		//}
		Rect box(ret_1.x, ret_1.y, ret_1.width, ret_1.height);
		rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
		drawContours(image, contours, maxIndex, Scalar(0, 255, 0), 2, 8, hierarchy);
	}
	double dist = pr.get_distance(known_W, ret_1.width);
	string dist_str = pr.Convert(dist);
	putText(image, "Distance:" + dist_str + "cm", Point(50, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(50, 250, 50), 2, 8);
	namedWindow("detected", WINDOW_FREERATIO);
	imshow("detected", image);
}