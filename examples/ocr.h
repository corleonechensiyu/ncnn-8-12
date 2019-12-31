#include <stdio.h>
#include <vector>
#include "platform.h"
#include "net.h"
#include <iostream>
#include <string>
#include "benchmark.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cpu.h>
#include <unistd.h>
using namespace std;

static int detect(const cv::Mat& bgr , vector<float>& scores)
{	
	
	ncnn::Net ocr;
        ocr.load_param("dd-v3.param");
        ocr.load_model("dd-v3.bin");
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data,ncnn::Mat::PIXEL_BGR,bgr.cols,bgr.rows,64,32);
	
        const float mean_vals[3]={141.f,135.f,122.f};
	in.substract_mean_normalize(mean_vals, 0);
	
	double start = ncnn::get_current_time();
	ncnn::Extractor ex = ocr.create_extractor();
	ex.set_num_threads(2);
	//ncnn::set_cpu_powersave(2);
	ex.input("data",in);

	ncnn::Mat out;
	ex.extract("result",out);
	double end = ncnn::get_current_time();
	double time = end - start;
	static const char* ocr_names[]={"blank","0","1","2","3","4","5","6","7","8","9","."};
	string str;

	scores.resize(5);
	for(int j=0;j<5;j++)
	{
		//scores[j]=out[j];
	
		if(0<=out[j]-1 && out[j]<12)
		{
			scores[j]=out[j];
			int index=scores[j];
			str += ocr_names[index];
		}
	}
	cout<<str<<endl;
	cout<<time<<endl;




	return 0;

}


