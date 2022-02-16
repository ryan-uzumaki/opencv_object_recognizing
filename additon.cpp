#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "opencv2/objdetect.hpp"
#include "stdlib.h"
#include "core/core.hpp"  
#include "highgui/highgui.hpp"  
#include "imgproc/imgproc.hpp"
#include <algorithm>

using namespace std;
using namespace cv;

int main2() {
	VideoCapture capture(0);
	Mat frame;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		imshow("detected", frame);
		Mat dst;
		bilateralFilter(frame, dst, 5, 20, 20);
		cvtColor(dst, dst, COLOR_BGR2HSV);
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
		Mat m_ResImg;
		erode(dst, m_ResImg, element);//进行腐蚀操作
		Mat mask;
		inRange(m_ResImg, Scalar(35, 43, 46), Scalar(77, 255, 255),mask);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(m_ResImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
		double cnts;
		cnts = contourArea(contours);
		RotatedRect rrt = minAreaRect(cnts);
		Mat pts;
		boxPoints(rrt, pts);
		drawContours(frame, contours, 0, Scalar(0, 0, 255), -1, 8);
		int c = waitKey(1);
		if (c == 27) { // 退出
			break;
		}
	}
	return 0;
}
