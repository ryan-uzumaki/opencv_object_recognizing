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
		if (c == 27) { // ÍË³ö
			break;
		}
	}
	//capture.release();
	return 0;
}







