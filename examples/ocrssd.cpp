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
#include "ocr.h"
#include <string.h>
#include <iostream>
using namespace std;
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};


/*static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background","text"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        //        obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("MSSD", image);
    
}*/

static void recog(const cv::Mat& bgr, const std::vector<Object>& objects)
{
	cv::Mat image = bgr.clone();
	for (size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];
		vector<float> scores;
		if(obj.rect.x>300)
		{
			cv::Mat roi = image(obj.rect).clone();
			detect(roi,scores);
			//cv::imwrite(name+to_string(i)+".jpg",roi);
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
	//CPU_SET(2,&mask);
	//CPU_SET(3,&mask);
	if(sched_setaffinity(0,sizeof(cpu_set_t),&mask)<0)
	{
		printf("Error: setaffinity()\n");
		exit(0);
	}
	//cv::VideoCapture capture(0);
	cv::VideoCapture capture("test.mp4");
	capture.set(CV_CAP_PROP_FRAME_WIDTH,640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);
	cv::namedWindow("MSSD",CV_WINDOW_NORMAL);
	cvResizeWindow("MSSD",640,480);
    //int num=0;
	vector<cv::Mat> save;
	save.resize(15);
	while(1)
	{
		
		//++num;
		//string name ="test/67_"+to_string(num);
		pthread_mutex_lock(&m_frame);
		capture >> frame;
		
		pthread_mutex_lock(&m_objects);
		recog(frame,objects);
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
	//const int target_size = 300;
	/*pthread_mutex_lock(&m_frame);
	int img_w = frame.cols;
	int img_h = frame.rows;
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data,ncnn::Mat::PIXEL_BGR,frame.cols,frame.rows,target_size,target_size);
	const float mean_vals[3]={127.5f,127.5f,127.5f};
	const float norm_vals[3]={1.0/127.5,1.0/127.5,1.0/127.5};
	in.substract_mean_normalize(mean_vals,norm_vals);
	pthread_mutex_unlock(&m_frame);*/


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
        //const float norm_vals[3]={1.0/127.5,1.0/127.5,1.0/127.5};
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
    /*cv::Mat m = cv::imread("./ssd_horse.jpg", 1);
    if (m.empty())
    {
		printf("can't load image");
        return -1;
    }*/


    //std::vector<Object> objects;
    //detect_mobilenet(m, objects);


    //draw_objects(m, objects);
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
