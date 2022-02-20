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
#include "Process.hpp"
#include <iterator>


using namespace std;
using namespace cv;


//main entrance
int main() {
	VideoCapture capture(0);
	VideoCapture capture_1(0);
	Mat frame;
	Mat frame_1;
	Process object;
	while (true) {
		capture.read(frame);
		waitKey(1);
		capture_1.read(frame_1);
		if (frame.empty() || frame_1.empty()) {
			break;
		}
		object.object_recognition(frame,frame_1);
		int c = waitKey(1);
		if (c == 27) { // 退出
			break;
		}
	}
	//capture.release();
	return 0;
}







void detect_object(Mat& imageSource) {
	//imshow("Source Image", imageSource);
	Mat image = Mat::zeros(imageSource.size(), imageSource.type());
	image = imageSource.clone();
	//GaussianBlur(imageSource, image, Size(3, 3), 0);
	Canny(image, image, 50, 100);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);
	Mat Contours = Mat::zeros(image.size(), CV_8UC1);  //绘制
	for (int i = 0; i < contours.size(); i++) {
		//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
		for (int j = 0; j < contours[i].size(); j++) {
			//绘制出contours向量内所有的像素点
			Point P = Point(contours[i][j].x, contours[i][j].y);
			Contours.at<uchar>(P) = 255;
		}

		//输出hierarchy向量内容
		/*char ch[256];
		sprintf_s(ch, "%d", i);
		string str = ch;
		cout << "向量hierarchy的第" << str << " 个元素内容为：" << endl << hierarchy[i] << endl << endl;*/

		//绘制轮廓
		drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
	}
	imshow("Contours Image", imageContours); //轮廓
	//imshow("Point of Contours", Contours);   //向量contours内保存的所有轮廓点集
	waitKey(0);
}

