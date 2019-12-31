#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	VideoCapture capture("test.mp4");
	if(!capture.isOpened())
	{
		printf("error");
		return -1;
	}
	
	int frame_cnt=0;
	int delay =33;
	//vector<Mat> save;
	//save.resize(15);
	//string a[]={"1.jpg","2.jpg","3.jpg","4.jpg","5.jpg","6.jpg","7.jpg","8.jpg","9.jpg","10.jpg"};
	int i =0;
	while(1)
	{
		
		Mat frame;
		capture >>frame;
		imshow("show",frame);
		int key;
		key = waitKey(delay);
		if((char)key ==27)
			break;
		
		//1s 3 pics
		if(frame_cnt %2==0)	
		{
			++i;
			string name = "test/"+ to_string(i)+".jpg";
			imwrite(name,frame);
			//save.push_back(frame);
			
		}
		
		//cout<<save.size()<<"  "<<frame_cnt<<endl;
		frame_cnt++;
		//if(frame_cnt%30==0)
        //	save.clear();

	}
	capture.release();
	return 0;
}
