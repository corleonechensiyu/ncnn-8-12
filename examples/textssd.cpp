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
#include "benchmark.h"
#include "platform.h"
#include "net.h"
#include "cpu.h"
#include <unistd.h>
#include <pthread.h>
#include <iostream>
#include <sstream>
#include "ocr.h"
using namespace std;
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_mobilenet(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net mobilenet;
	//mobilenet.opt.use_packing_layout=true;
    mobilenet.load_param("jjl-v2.param");
    mobilenet.load_model("jjl-v2.bin");

    //const int target_size = 300;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows,320 ,180);

    //const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    //const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    //in.substract_mean_normalize(mean_vals, norm_vals);
        const float mean_vals[3] = {141.f,135.f,122.f};
	in.substract_mean_normalize(mean_vals, 0);
    ncnn::Extractor ex = mobilenet.create_extractor();

	ex.set_num_threads(2);
	//ncnn::set_cpu_powersave(1);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out",out);

//     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i=0; i<out.h; i++)
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

    return 0;
}

static void recog(const cv::Mat& bgr, const std::vector<Object>& objects)
{
	cv::Mat image = bgr.clone();
	for (size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];
		vector<float> scores;
		if(obj.rect.x>20)
		{
			cv::Mat roi = image(obj.rect).clone();
			detect(roi,scores);
		}
	}
}


int main()
{
    
    cv::Mat m = cv::imread("000013.jpg", 1);
    if (m.empty())
    {
		printf("can't load image");
        return -1;
    }


	
    std::vector<Object> objects;
	double start = ncnn::get_current_time();
    detect_mobilenet(m, objects);
    recog(m, objects);
	double end = ncnn::get_current_time();
	double time = end - start;
	cout<<time<<"ms"<<endl;
    return 0;
}
