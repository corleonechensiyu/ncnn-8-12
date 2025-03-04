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

#include "platform.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN
#include "benchmark.h"
#include <iostream>
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_squeezenet(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net shufflenet;

#if NCNN_VULKAN
    squeezenet.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    // original pretrained model from https://github.com/chuanqi305/SqueezeNet-SSD
    // squeezenet_ssd_voc_deploy.prototxt
    // https://drive.google.com/open?id=0B3gersZ2cHIxdGpyZlZnbEQ5Snc
    shufflenet.load_param("jjl-v2.param");
    shufflenet.load_model("jjl-v2.bin");

    //const int target_size = 300;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 320, 180);

    //const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
	//const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
	const float mean_vals[3] = {141.f,135.f,122.f};
    in.substract_mean_normalize(mean_vals,0);

    ncnn::Extractor ex = shufflenet.create_extractor();
    ex.set_num_threads(4);
	double start = ncnn::get_current_time();
    ex.input("data", in);

    ncnn::Mat out;
	ncnn::Mat conv;
	ex.extract("conv3_3_norm_mbox_loc_new",conv);
    ex.extract("detection_out",out);
	double end = ncnn::get_current_time();
	double time = end -start;
	std::cout<<"time: "<<time<<std::endl;
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

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

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

    cv::imshow("image", image);
    cv::waitKey(0);
}

//int main(int argc, char** argv)
int main()
{
   /* if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }*/
	cv::Mat m = cv::imread("000013.jpg", 1);

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN

    std::vector<Object> objects;
    detect_squeezenet(m, objects);

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

    draw_objects(m, objects);

    return 0;
}
