#include "core/core.hpp"  
#include "highgui/highgui.hpp"  
#include "imgproc/imgproc.hpp"  
#include "iostream"

using namespace std;
using namespace cv;

void detect_object1(Mat & imageSource){
	imshow("Source Image", imageSource);
	Mat image;
	GaussianBlur(imageSource, image, Size(3, 3), 0);
	Canny(image, image, 100, 250);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);
	Mat Contours = Mat::zeros(image.size(), CV_8UC1);  //����
	for (int i = 0; i < contours.size(); i++){
		//contours[i]������ǵ�i��������contours[i].size()������ǵ�i�����������е����ص���
		for (int j = 0; j < contours[i].size(); j++){
			//���Ƴ�contours���������е����ص�
			Point P = Point(contours[i][j].x, contours[i][j].y);
			Contours.at<uchar>(P) = 255;
		}

		//���hierarchy��������
		/*char ch[256];
		sprintf_s(ch, "%d", i);
		string str = ch;
		cout << "����hierarchy�ĵ�" << str << " ��Ԫ������Ϊ��" << endl << hierarchy[i] << endl << endl;*/

		//��������
		drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
	}
	imshow("Contours Image", imageContours); //����
	//imshow("Point of Contours", Contours);   //����contours�ڱ�������������㼯
	waitKey(0);
}