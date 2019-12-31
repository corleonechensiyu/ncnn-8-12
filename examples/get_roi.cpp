// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "platform.h"
#include "net.h"
#include <unistd.h>
#include <pthread.h>
#include "cpu.h"
#include "benchmark.h"

#include <string.h>
#include <iostream>
using namespace std;
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static void recog(const cv::Mat& bgr, const std::vector<Object>& objects,string& name )
{
	cv::Mat image = bgr.clone();
	for (size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];
		//vector<float> scores;
		//cv::Mat roi = image(obj.rect).clone();
		if(obj.rect.x>0 && obj.rect.y>0)
		{
			if(obj.rect.x>320 && obj.rect.x>100)		
			{
				cv::Mat roi = image(obj.rect).clone();
				cv::imwrite(name+to_string(i)+".jpg",roi);
			}
		}
		
	}
}

cv::Mat frame;
cv::Mat currentframe,preframe;
bool quit_flag=false;
std::vector<Object> objects;
pthread_mutex_t m_frame,m_quit,m_objects;

void *th_vedio(void *)
{
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(0,&mask);
	CPU_SET(1,&mask);
	
	if(sched_setaffinity(0,sizeof(cpu_set_t),&mask)<0)
	{
		printf("Error: setaffinity()\n");
		exit(0);
	}
	
	cv::VideoCapture capture("VID_20170112_143338.3gp");
	capture.set(CV_CAP_PROP_FRAME_WIDTH,640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);
	cv::namedWindow("MSSD",CV_WINDOW_NORMAL);
	cvResizeWindow("MSSD",640,480);

	vector<cv::Mat> save;
	save.resize(15);
	int num=0;
	while(1)
	{
		
		++num;
		string name ="test/7.9_"+to_string(num);
		pthread_mutex_lock(&m_frame);
		capture >> frame;
		cv::flip(frame,frame,-1);
		pthread_mutex_lock(&m_objects);
		recog(frame,objects,name);
		pthread_mutex_unlock(&m_objects);
				
		cv::imshow("MSSD",frame);
		pthread_mutex_unlock(&m_frame);
		if(cv::waitKey(10)=='q')
		{
			pthread_mutex_lock(&m_quit);
			quit_flag = true;
			pthread_mutex_unlock(&m_quit);
			break;
		}		
		usleep(50000);
	}
	return 0;
}

void *th_detect(void*)
{
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(2,&mask);
	CPU_SET(3,&mask);
	if(sched_setaffinity(0,sizeof(cpu_set_t),&mask)<0)
	{
		printf("Error: setaffinity()\n");
		exit(0);
	}
	ncnn::Net mobilenet;
	mobilenet.load_param("jjl-v1.param");
	mobilenet.load_model("jjl-v1.bin");

	while(1)
	{
		pthread_mutex_lock(&m_quit);
		if(quit_flag) break;
		pthread_mutex_unlock(&m_quit);

		pthread_mutex_lock(&m_frame);
        int img_w = frame.cols;
        int img_h = frame.rows;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data,ncnn::Mat::PIXEL_BGR,frame.cols,frame.rows,320,240);
        const float mean_vals[3]={104.f,117.f,123.f};
        
        in.substract_mean_normalize(mean_vals,0);
        pthread_mutex_unlock(&m_frame);

		ncnn::Extractor ex = mobilenet.create_extractor();
		ex.set_num_threads(2);
		ncnn::set_cpu_powersave(2);
		ex.input("data",in);
		ncnn::Mat out;
		ex.extract("detection_out",out);
		
		objects.clear();
		pthread_mutex_lock(&m_objects);
		for(int i=0;i<out.h;i++)
		{
			const float* values = out.row(i);
			Object object;
            object.label = values[0];
		    object.prob = values[1];
            object.rect.x = values[2] * img_w;
            object.rect.y = values[3] * img_h;
            object.rect.width = values[4] * img_w - object.rect.x;
            object.rect.height = values[5] * img_h - object.rect.y;
  
            objects.push_back(object);

		}
		pthread_mutex_unlock(&m_objects);
	}
	return 0;
}

int main()
{
    
	pthread_mutex_init(&m_frame,NULL);
	pthread_mutex_init(&m_quit,NULL);
	pthread_mutex_init(&m_objects,NULL);

	pthread_t id1,id2;
	pthread_create(&id1,NULL,th_vedio,NULL);
	pthread_create(&id2,NULL,th_detect,NULL);

	pthread_join(id1,NULL);
	pthread_join(id2,NULL);

	pthread_mutex_destroy(&m_frame);
	pthread_mutex_destroy(&m_quit);
	pthread_mutex_destroy(&m_objects);

    return 0;
}
