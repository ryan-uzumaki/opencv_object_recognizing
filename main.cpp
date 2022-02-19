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


using namespace std;
using namespace cv;
int known_W = 9;
int known_P = 510;

double get_distance(int W, int P);
string Convert(float Num);


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


int main() {
	VideoCapture capture(0);
	Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		Mat temp = Mat::zeros(frame.size(), frame.type());
		Mat m = Mat::zeros(frame.size(), frame.type());
		addWeighted(frame, 0.19, m, 0.0, 0, temp);
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
		size_t maxIndex = argmax(area.begin(), area.end());
		Rect ret_1 = boundingRect(contours[maxIndex]);
		int avgX, avgY;
		avgX = (ret_1.x + ret_1.width) / 2;
		avgY = (ret_1.y + ret_1.height) / 2;
		for (int i = 0; i < contours.size(); i++) {
			for (int j = 0; j < contours[i].size(); j++) {
				Point P = Point(contours[i][j].x, contours[i][j].y);
				Mat Contours = Mat::zeros(m_ResImg.size(), CV_8UC1);  //绘制
				Contours.at<uchar>(P) = 255;
			}
			Rect box(ret_1.x, ret_1.y, ret_1.width, ret_1.height);
			rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
			drawContours(frame, contours, maxIndex, Scalar(0, 255, 0), 2, 8, hierarchy);
		}
		double dist = get_distance(known_W, ret_1.width);
		string dist_str = Convert(dist);
		putText(frame, "Distance:" + dist_str + "cm", Point(50, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(50, 250, 50), 2, 8);
		namedWindow("detected", WINDOW_FREERATIO);
		imshow("detected", frame);
		int c = waitKey(1);
		if (c == 27) { // 退出
			break;
		}
	}
	capture.release();
	return 0;
}


string Convert(float Num)
{
	std::ostringstream oss;
	oss << Num;
	std::string str(oss.str());
	return str;
}


double get_distance(int W, int P) {
	double F = 550;
	double D = 0;
	D = (W * F) / P;
	return D;
}
