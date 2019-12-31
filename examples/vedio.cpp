#include <opencv2/opencv.hpp>
#include <stdio.h>
using namespace std;
using namespace cv;

int main()
{
	VideoCapture capture(0);
	if(!capture.isOpened())
	{
		printf("open video failed\n");
		return 1;
	}
	Mat frame;
	int frameNum=5000;
	string outputVidelPath="test.mp4";

	cv::Size swh = cv::Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH),(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));

	VideoWriter outputVideo;
	outputVideo.open(outputVidelPath,CV_FOURCC('M','P','4','2'),25.0,swh);

	while(capture.isOpened() && frameNum>0)
	{
		capture >>frame;
		if(frame.empty()) break;
		outputVideo <<frame;

		frameNum --;

		imshow("img",frame);
		waitKey(0);
		if(char(waitKey(1)=='q')) break;
	}
	
	outputVideo.release();
	system("pause");


	
}
