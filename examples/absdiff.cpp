#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <wiringPi.h>
using namespace std;
using namespace cv;
//int num=0;
int main()
{
	wiringPiSetup();
	pinMode(7,OUTPUT);
	VideoCapture capture("test.mp4");
	if(!capture.isOpened())
	{
		printf("open video failed\n");
		return 1;
	}
	Mat frame;
	Mat currframe,preframe;
	Mat premeanvalue,premeanstd,currmeanvalue,currmeanstd;
	while(true)
	{
		capture >> frame;
		imshow("video",frame);
		preframe=frame.clone();
		
		capture >> frame;
		currframe = frame.clone();

		cvtColor(preframe,preframe,CV_BGR2GRAY);
		cvtColor(currframe,currframe,CV_BGR2GRAY);
		meanStdDev(preframe,premeanvalue,premeanstd);
		meanStdDev(currframe,currmeanvalue,currmeanstd);
		double pre=premeanstd.at<double>(0,0);
		double current=currmeanstd.at<double>(0,0);
		double abss=abs(pre-current);
		if(abss>3.0){
			cout<<"changed"<<endl;
			digitalWrite(7,HIGH);
			delay(500);
			digitalWrite(7,LOW);
			delay(500);
		}
		//cout<<"abs: "<<abss<<endl;
		//cout<<"currentimage: "<<current<<endl;
		//threshold(diffframe,tempframe,20,255.0,CV_THRESH_BINARY);
		/*dilate(tempframe,tempframe,Mat());
		erode(tempframe,tempframe,Mat());	
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(tempframe,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,Point());
		
		for(int i=0;i<contours.size();i++)
		{
			double area = contourArea(contours[i]);
			if(area<1000){
				continue;
			}
			else
			{
				printf("11111111111\n");
			}
		}*/


		
		waitKey(33);
		

	}
		

		
		
		
	
	


	return 0;
}
