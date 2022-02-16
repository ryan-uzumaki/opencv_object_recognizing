
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "opencv2/objdetect.hpp"
#include "stdlib.h"
#include "core/core.hpp"  
#include "highgui/highgui.hpp"  
#include "imgproc/imgproc.hpp"  

using namespace std;
using namespace cv;

void detect_object(Mat& imageSource) {
	//imshow("Source Image", imageSource);
	Mat image=Mat::zeros(imageSource.size(),imageSource.type());
	image=imageSource.clone();
	//GaussianBlur(imageSource, image, Size(3, 3), 0);
	Canny(image, image, 50, 100);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);
	Mat Contours = Mat::zeros(image.size(), CV_8UC1);  //����
	for (int i = 0; i < contours.size(); i++) {
		//contours[i]������ǵ�i��������contours[i].size()������ǵ�i�����������е����ص���
		for (int j = 0; j < contours[i].size(); j++) {
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


//int main1(int, char* argv[])
//{
//	Mat OriginalImg;
//
//	OriginalImg = imread("C:\\Users\\ryand\\Desktop\\test_detecting.jpg", IMREAD_COLOR);//��ȡԭʼ��ɫͼ��
//	if (OriginalImg.empty())  //�ж�ͼ��Է��ȡ�ɹ�
//	{
//		cout << "����!��ȡͼ��ʧ��\n";
//		return -1;
//	}
//	//imshow("ԭͼ", OriginalImg); //��ʾԭʼͼ��
//	cout << "Width:" << OriginalImg.rows << "\tHeight:" << OriginalImg.cols << endl;//��ӡ����
//
//	Mat ResizeImg;
//	if (OriginalImg.cols > 640) {
//		resize(OriginalImg, ResizeImg, Size(640, 640 * OriginalImg.rows / OriginalImg.cols));
//	}
//	imshow("�ߴ�任ͼ", ResizeImg);
//
//	unsigned char pixelB, pixelG, pixelR;  //��¼��ͨ��ֵ
//	unsigned char DifMax = 50;             //������ɫ���ֵ���ֵ����
//	unsigned char B = 138, G = 63, R = 23; //��ͨ������ֵ�趨���������ɫ����
//	Mat BinRGBImg = ResizeImg.clone();  //��ֵ��֮���ͼ��
//	int i = 0, j = 0;
//	for (i = 0; i < ResizeImg.rows; i++)   //ͨ����ɫ������ͼƬ���ж�ֵ������
//	{
//		for (j = 0; j < ResizeImg.cols; j++)
//		{
//			pixelB = ResizeImg.at<Vec3b>(i, j)[0]; //��ȡͼƬ����ͨ����ֵ
//			pixelG = ResizeImg.at<Vec3b>(i, j)[1];
//			pixelR = ResizeImg.at<Vec3b>(i, j)[2];
//
//			if (abs(pixelB - B) < DifMax && abs(pixelG - G) < DifMax && abs(pixelR - R) < DifMax)
//			{                                           //������ͨ����ֵ�͸���ͨ����ֵ���бȽ�
//				BinRGBImg.at<Vec3b>(i, j)[0] = 255;     //������ɫ��ֵ��Χ�ڵ����óɰ�ɫ
//				BinRGBImg.at<Vec3b>(i, j)[1] = 255;
//				BinRGBImg.at<Vec3b>(i, j)[2] = 255;
//			}
//			else
//			{
//				BinRGBImg.at<Vec3b>(i, j)[0] = 0;        //��������ɫ��ֵ��Χ�ڵ�����Ϊ��ɫ
//				BinRGBImg.at<Vec3b>(i, j)[1] = 0;
//				BinRGBImg.at<Vec3b>(i, j)[2] = 0;
//			}
//		}
//	}
//	imshow("������ɫ��Ϣ��ֵ��", BinRGBImg);        //��ʾ��ֵ������֮���ͼ��
//
//	Mat BinOriImg;     //��̬ѧ������ͼ��
//	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3)); //������̬ѧ�����Ĵ�С
//	dilate(BinRGBImg, BinOriImg, element);     //���ж�����Ͳ���
//	dilate(BinOriImg, BinOriImg, element);
//	dilate(BinOriImg, BinOriImg, element);
//
//	erode(BinOriImg, BinOriImg, element);      //���ж�θ�ʴ����
//	erode(BinOriImg, BinOriImg, element);
//	erode(BinOriImg, BinOriImg, element);
//	imshow("��̬ѧ�����", BinOriImg);        //��ʾ��̬ѧ����֮���ͼ��
//
//	double length, area, rectArea;     //���������ܳ�����������������
//	double rectDegree = 0.0;           //���ζ�=���������/�������
//	double long2Short = 0.0;           //��̬��=����/�̱�
//	CvRect rect;           //������
//	CvBox2D box, boxTemp;  //��Ӿ���
//	CvPoint2D32f pt[4];    //���ζ������
//	double axisLong = 0.0, axisShort = 0.0;        //���εĳ��ߺͶ̱�
//	double axisLongTemp = 0.0, axisShortTemp = 0.0;//���εĳ��ߺͶ̱�
//	double LengthTemp;     //�м����
//	float  angle = 0;      //��¼���Ƶ���б�Ƕ�
//	float  angleTemp = 0;
//	bool   TestPlantFlag = 0;  //���Ƽ��ɹ���־λ
//	cvtColor(BinOriImg, BinOriImg, CV_BGR2GRAY);   //����̬ѧ����֮���ͼ��ת��Ϊ�Ҷ�ͼ��
//	threshold(BinOriImg, BinOriImg, 100, 255, THRESH_BINARY); //�Ҷ�ͼ���ֵ��
//	//detect_object(BinOriImg);
//	vector<vector<Point>> contours;
//	vector<Vec<int,4>> hierarchy;
//	//CvMemStorage* storage = cvCreateMemStorage(0);
//	//CvSeq* seq = 0;     //����һ������,CvSeq�������һ���������������У����ǹ̶�������
//	//CvSeq* tempSeq = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
//	findContours(BinOriImg, contours, hierarchy, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
//	int cnt = size(contours);
//	//int cnt = cvFindContours(&(IplImage(BinOriImg)), storage, &seq, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//	//��һ��������IplImageָ�����ͣ���MATǿ��ת��ΪIplImageָ������
//	//������������Ŀ 
//	//��ȡ��ֵͼ���������ĸ���
//	cout << "number of contours:" << cnt << endl;  //��ӡ��������
//	for (tempSeq = seq; tempSeq != NULL; tempSeq = tempSeq->h_next)
//	{
//		length = cvArcLength(tempSeq);       //��ȡ�����ܳ�
//		area = cvContourArea(tempSeq);       //��ȡ�������
//		if (area > 800 && area < 50000)     //�������������С�ж�
//		{
//			rect = cvBoundingRect(tempSeq, 1);//������α߽�
//			boxTemp = cvMinAreaRect2(tempSeq, 0);  //��ȡ�����ľ���
//			cvBoxPoints(boxTemp, pt);              //��ȡ�����ĸ���������
//			angleTemp = boxTemp.angle;                 //�õ�������б�Ƕ�
//
//			axisLongTemp = sqrt(pow(pt[1].x - pt[0].x, 2) + pow(pt[1].y - pt[0].y, 2));  //���㳤�ᣨ���ɶ���
//			axisShortTemp = sqrt(pow(pt[2].x - pt[1].x, 2) + pow(pt[2].y - pt[1].y, 2)); //������ᣨ���ɶ���
//
//			if (axisShortTemp > axisLongTemp)   //������ڳ��ᣬ��������
//			{
//				LengthTemp = axisLongTemp;
//				axisLongTemp = axisShortTemp;
//				axisShortTemp = LengthTemp;
//			}
//			else
//				angleTemp += 90;
//			rectArea = axisLongTemp * axisShortTemp;  //������ε����
//			rectDegree = area / rectArea;     //������ζȣ���ֵԽ�ӽ�1˵��Խ�ӽ����Σ�
//
//			long2Short = axisLongTemp / axisShortTemp; //���㳤���
//			if (long2Short > 2.2 && long2Short < 3.8 && rectDegree > 0.63 && rectDegree < 1.37 && rectArea > 2000 && rectArea < 50000)
//			{
//				Mat GuiRGBImg = ResizeImg.clone();
//				TestPlantFlag = true;             //��⳵������ɹ�
//				for (int i = 0; i < 4; ++i) {
//					line((Mat(GuiRGBImg)), cvPointFrom32f(pt[i]), cvPointFrom32f(pt[((i + 1) % 4) ? (i + 1) : 0]), CV_RGB(255, 0, 0));
//				}       //���߿����������
//				imshow("��ȡ���ƽ��ͼ", GuiRGBImg);    //��ʾ���ս��ͼ
//
//				box = boxTemp;
//				angle = angleTemp;
//				axisLong = axisLongTemp;
//				axisShort = axisShortTemp;
//				cout << "��б�Ƕȣ�" << angle << endl;
//			}
//		}
//	}
//
//	waitKey();
//	return 0;
//
//}

int main() {
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
		Mat m_ResImg;
		cvtColor(dst, m_ResImg, COLOR_BGR2HSV);
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
		erode(m_ResImg, m_ResImg, element);//���и�ʴ����
		Mat mask;
		inRange(m_ResImg, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);
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
		if (c == 27) { // �˳�
			break;
		}
	}
	return 0;
}


