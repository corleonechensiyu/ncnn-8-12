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
#include <iostream>
#include "platform.h"
#include "net.h"
#include "layer.h"
#include "layer_type.h"
#include <math.h>
#include <functional>
#include <tuple>
#include "benchmark.h"
#include <algorithm>
#include <iterator>
#include <math.h>
struct Object
{
    std::vector<cv::Point_<float>> point;
    std::vector<float> scores;
};
static void sig(ncnn::Mat& out)
{
    ncnn::Layer* sigmoid = ncnn::create_layer(ncnn::LayerType::Sigmoid);
    sigmoid->forward_inplace(out);
    delete sigmoid;

}
static void concatenate(const ncnn::Mat& out,const ncnn::Mat& out1,const ncnn::Mat& out2 ,ncnn::Mat& result)
{
    ncnn::Layer* concat = ncnn::create_layer(ncnn::LayerType::Concat);
    ncnn::ParamDict pd;
    pd.set(0,0);
    concat->load_param(pd);
    //forword
    std::vector<ncnn::Mat> bottoms(3);
    bottoms[0] = out;
    bottoms[1] = out1;
    bottoms[2] = out2;
    std::vector<ncnn::Mat> tops(1);
    concat->forward(bottoms,tops);
    result = tops[0];
    delete concat;
}


static void region_neighbor(const std::vector<std::tuple<int,int>>& region_list,std::vector<std::tuple<int,int>>& neighbor)
{
    std::vector<int> n_tuple_y;
    std::vector<int>::iterator it_max;
    std::vector<int>::iterator it_min;
    int j_min;
    int j_max;
    int i_m = std::get<0>(region_list[0]) +1;
    for (size_t i = 0; i < region_list.size(); i++)
    {
        n_tuple_y.push_back(std::get<1>(region_list[i]));
        auto new_n = std::make_tuple(i_m,std::get<1>(region_list[i]));
        neighbor.push_back(new_n);
    }
    it_max = max_element(n_tuple_y.begin(),n_tuple_y.end());
    it_min = min_element(n_tuple_y.begin(),n_tuple_y.end());
    j_min = *it_min -1;
    j_max = *it_max +1;
    auto bar1 = std::make_tuple(i_m,j_min);
    auto bar2 = std::make_tuple(i_m,j_max);
    neighbor.push_back(bar1);
    neighbor.push_back(bar2);

}

static int rec_region_merge(const std::vector<std::vector<std::tuple<int,int>>>& region_lists,const int & m ,std::vector<int>& S,std::vector<int>& s_D)
{
    std::vector<int> rows ={m};
    std::vector<int> tmp;
    tmp.clear();
    s_D.push_back(m);
    for (std::vector<int>::iterator it = S.begin(); it != S.end(); ++it)
    {
        std::vector<std::tuple<int,int>> neighbor1;
        std::vector<std::tuple<int,int>> neighbor2;
        std::vector<std::tuple<int,int>> mid1;
        std::vector<std::tuple<int,int>> mid2;
        region_neighbor(region_lists[m],neighbor1);
        region_neighbor(region_lists[*it],neighbor2);
        std::set_intersection(neighbor1.begin(),neighbor1.end(),region_lists[*it].begin(),region_lists[*it].end(),std::inserter(mid1,mid1.begin()));
        std::set_intersection(neighbor2.begin(),neighbor2.end(),region_lists[m].begin(),region_lists[m].end(),std::inserter(mid2,mid2.begin()));
        if (!mid1.empty() || !mid2.empty())
        {
            tmp.push_back(*it);
        }
    }
    for (size_t i : tmp)
    {
        S.erase(remove(S.begin(),S.end(),i),S.end());//删除S向量中指定值
    }
    for (size_t j : tmp)
    {
        rec_region_merge(region_lists,j,S,s_D);
    }
    return 0;
}


static void nms(const ncnn::Mat& result,std::vector<std::tuple<int,int>>& triples,const int & h,std::vector<Object>& objects)
{
    const float threshold = 0.9;
    const int pixel_size =4;
    const float epsilon = 1e-4;
    std::vector<int> S;
    std::vector<std::vector<int>> D;
    std::vector<std::vector<std::tuple<int,int>>> region_lists;
    region_lists.clear();
    for(size_t i=0;i<triples.size();i++)
    {
        int x = std::get<0>(triples[i]);
        int y = std::get<1>(triples[i]);
        bool merge = false;
        std::vector<std::tuple<int,int>> region_list;
        std::vector<std::tuple<int,int>> region_neighbor;
        std::vector<std::tuple<int,int>> mid;
        region_list.clear();
        region_list.push_back(triples[i]);
        auto neighbor =std::make_tuple(x,y-1);
        region_neighbor.push_back(neighbor);
        //isdisjoint
        for(size_t k = 0; k < region_lists.size(); k++)
        {
            std::set_intersection(region_neighbor.begin(),region_neighbor.end(),region_lists[k].begin(),region_lists[k].end(),std::inserter(mid,mid.begin()));
            if (!mid.empty())
            {
                region_lists[k].push_back(triples[i]);
                merge = true;
                mid.clear();
            }
        }
        if(!merge)
        {
            region_lists.push_back(region_list);
            region_list.clear();
        }
    }

    
    for(size_t q=0;q<region_lists.size();q++)
    {
        S.push_back(q);
    }
    while (S.size()>0)
    {
        int m =S[0];
        S.erase(S.begin());
        if (S.size()==0)
        {
            D[D.size()-1].push_back(m);
        }
        else
        {
            std::vector<int> s_D;
            rec_region_merge(region_lists,m,S,s_D);
            D.push_back(s_D);
        }    
    }
    int D_size = D.size();
    const float* ptr1 = result.channel(1);
    const float* ptr2 = result.channel(2);
    const float* ptr3 = result.channel(3);
    const float* ptr4 = result.channel(4);
    const float* ptr5 = result.channel(5);
    const float* ptr6 = result.channel(6);
    float quad_list[D_size][4][2] ={0.0f};
    float score_list[D_size][4] ={0.0f};
    for (int k = 0; k < D_size; k++)
    {
        Object object;
        float total_score[4][2] = {0.0f};
        std::vector<int> group;
        group = D[k];
        for (int row : group)
        {
            for (std::tuple<int,int> ij : region_lists[row])
            {
                int x = std::get<0>(ij);
                int y = std::get<1>(ij);
                float score = ptr1[x*h+y];
                if (score >= threshold)
                {
                   float ith_score = ptr2[x*h+y];
                   if (ith_score< 0.1 || ith_score >=0.9) 
                   {
                        float p_v[2][2] ={0.0f};
                        int ith = round(ith_score);

                        total_score[ith*2][0] += score;
                        total_score[ith*2][1] += score;
                        total_score[ith*2+1][0] += score;
                        total_score[ith*2+1][1] += score;
                        int px = (y + 0.5) * pixel_size;
                        int py = (x + 0.5) * pixel_size;
                        p_v[0][0] = ptr3[x*h+y] + px;
                        p_v[0][1] = ptr4[x*h+y] + py;
                        p_v[1][0] = ptr5[x*h+y] + px;
                        p_v[1][1] = ptr6[x*h+y] + py;
                        quad_list[k][ith*2][0] += score *p_v[0][0];
                        quad_list[k][ith*2][1] += score *p_v[0][1];
                        quad_list[k][ith*2+1][0] += score *p_v[1][0];
                        quad_list[k][ith*2+1][1] += score *p_v[1][1];
                   }  
                }                
            }            
        }
        for (int s = 0; s < 4; s++)
        {
            score_list[k][s] =total_score[s][0];
            object.scores.push_back(score_list[k][s]);
        }
        for (int j = 0; j < 4; j++)
        {
            cv::Point_<float> p;
            for (int z = 0; z < 2; z++)
            {
                total_score[j][z] += epsilon;
                quad_list[k][j][z] = quad_list[k][j][z] /total_score[j][z];
            }
            p.x = quad_list[k][j][0];
            p.y = quad_list[k][j][1];
            object.point.push_back(p);
        }
        objects.push_back(object);
    }

}

static int detect_east(const cv::Mat& bgr, std::vector<Object>& objects)
{

    ncnn::Net east;
    east.load_param("advancedeast-nobn.param");
    east.load_model("advancedeast-nobn.bin");
    
    //const int target_size = 256;
    //ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_size, target_size);
    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);
    const float mean_vals[3] = {127.f, 127.f, 127.f};
    const float norm_vals[3] = {1.0/128.f,1.0/128.f,1.0/128.f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    double start = ncnn::get_current_time();
    ncnn::Extractor ex = east.create_extractor();
    ex.input("input_img", in);

    ncnn::Mat out;
    ncnn::Mat out1;
    ncnn::Mat out2;
    ex.extract("convolution_output",out);
    ex.extract("convolution_output1",out1);
    ex.extract("convolution_output2",out2);
    sig(out);
    sig(out1);
    ncnn::Mat result;
    concatenate(out,out1,out2,result);
    double zhong = ncnn::get_current_time();

    std::vector<std::tuple<int,int>> triples;
    int w = result.w;
    int h = result.h;
    //w h c pixel_threshold=0.9 np.greater_equal
    for(int q=0; q<1; q++)
    {
        const float* ptr = result.channel(q);
        for(int i=0 ;i<h; i++)
        {
            for(int j=0; j<w;j++)
            {
                if(ptr[i*h + j] >= 0.9)
                {
                    // index.push_back(i*h + j);
                    auto i_j = std::make_tuple(i,j);
                    triples.push_back(i_j);
                }
            }
        }
    }
    nms(result,triples,h,objects);
    double end = ncnn::get_current_time();
    double time1 = zhong - start;
    double time2 = end - start;
	std::cout<<time1<<std::endl;
	std::cout<<time2<<std::endl;

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{

   cv::Mat image = bgr.clone();

   for (size_t i = 0; i < objects.size(); i++)
   {
       const Object& obj = objects[i];
       std::vector<float> a = obj.scores;
       std::vector<float>::iterator it;
       it = min_element(a.begin(),a.end());
       if (*it > 0)
       {
           for (size_t j = 1; j < 4; j++)
           {
               cv::line(image,obj.point[j-1],obj.point[j],cv::Scalar(0,255,255),2);
           }
           cv::line(image,obj.point[3],obj.point[0],cv::Scalar(0,255,255),2);
       }
       
   }

   cv::imshow("image", image);
   cv::waitKey(0);
}

int main()
{

    cv::Mat m = cv::imread("013.jpg", 1);
    if (m.empty())
    {
        printf("no img");
        return -1;
    }

    std::vector<Object> objects;
    detect_east(m, objects);
    draw_objects(m, objects);

    return 0;
}
