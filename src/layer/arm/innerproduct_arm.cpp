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

#include "innerproduct_arm.h"
#include <iostream>
#include "benchmark.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
namespace ncnn {

DEFINE_LAYER_CREATOR(InnerProduct_arm)

int InnerProduct_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (use_int8_inference)
    {
        // TODO
        return InnerProduct::forward(bottom_blob, top_blob, opt);
    }
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;
    if(axis==1)
   {
	    top_blob.create(num_output, elemsize, opt.blob_allocator);
	    if (top_blob.empty())
		return -100;

	    const float* weight_data_ptr = weight_data;
	    int nn_num_output = num_output >> 2;
            int remain_num_output_start = nn_num_output << 2;
	    #pragma omp parallel for num_threads(opt.num_threads)
	    for (int pp=0; pp<nn_num_output; pp++)
	    {
		int p = pp * 4;

		float sum0 = 0.f;
		float sum1 = 0.f;
		float sum2 = 0.f;
		float sum3 = 0.f;

		if (bias_term)
		{
		    sum0 = bias_data[p];
		    sum1 = bias_data[p+1];
		    sum2 = bias_data[p+2];
		    sum3 = bias_data[p+3];
		}

		const float* w0 = weight_data_ptr + size * channels * p;
		const float* w1 = weight_data_ptr + size * channels * (p+1);
		const float* w2 = weight_data_ptr + size * channels * (p+2);
		const float* w3 = weight_data_ptr + size * channels * (p+3);

	#if __ARM_NEON
		float32x4_t _sum0 = vdupq_n_f32(0.f);
		float32x4_t _sum1 = vdupq_n_f32(0.f);
		float32x4_t _sum2 = vdupq_n_f32(0.f);
		float32x4_t _sum3 = vdupq_n_f32(0.f);
	#endif // __ARM_NEON

		// channels
		for (int q=0; q<channels; q++)
		{
		    const float* m = bottom_blob.channel(q);

	#if __ARM_NEON
		    int nn = size >> 2;
		    int remain = size & 3;
	#else
		    int remain = size;
	#endif // __ARM_NEON

	#if __ARM_NEON
		    for (; nn>0; nn--)
		    {
		        float32x4_t _m = vld1q_f32(m);

		        float32x4_t _w0 = vld1q_f32(w0);
		        _sum0 = vmlaq_f32(_sum0, _m, _w0);

		        float32x4_t _w1 = vld1q_f32(w1);
		        _sum1 = vmlaq_f32(_sum1, _m, _w1);

		        float32x4_t _w2 = vld1q_f32(w2);
		        _sum2 = vmlaq_f32(_sum2, _m, _w2);

		        float32x4_t _w3 = vld1q_f32(w3);
		        _sum3 = vmlaq_f32(_sum3, _m, _w3);
		        m += 4;
		        w0 += 4;
		        w1 += 4;
		        w2 += 4;
		        w3 += 4;
		    }
	#endif // __ARM_NEON
		    for (; remain>0; remain--)
		    {
		        sum0 += *m * *w0;
		        sum1 += *m * *w1;
		        sum2 += *m * *w2;
		        sum3 += *m * *w3;

		        m++;
		        w0++;
		        w1++;
		        w2++;
		        w3++;
		    }

		}

	#if __ARM_NEON
		float32x2_t _sum0ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
		float32x2_t _sum1ss = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
		float32x2_t _sum2ss = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
		float32x2_t _sum3ss = vadd_f32(vget_low_f32(_sum3), vget_high_f32(_sum3));

		float32x2_t _sum01ss = vpadd_f32(_sum0ss, _sum1ss);
		float32x2_t _sum23ss = vpadd_f32(_sum2ss, _sum3ss);

		sum0 += vget_lane_f32(_sum01ss, 0);
		sum1 += vget_lane_f32(_sum01ss, 1);
		sum2 += vget_lane_f32(_sum23ss, 0);
		sum3 += vget_lane_f32(_sum23ss, 1);

	#endif // __ARM_NEON

		if (activation_type == 1)
		{
		    sum0 = std::max(sum0, 0.f);
		    sum1 = std::max(sum1, 0.f);
		    sum2 = std::max(sum2, 0.f);
		    sum3 = std::max(sum3, 0.f);
		}
		else if (activation_type == 2)
		{
		    float slope = activation_params[0];
		    sum0 = sum0 > 0.f ? sum0 : sum0 * slope;
		    sum1 = sum1 > 0.f ? sum1 : sum1 * slope;
		    sum2 = sum2 > 0.f ? sum2 : sum2 * slope;
		    sum3 = sum3 > 0.f ? sum3 : sum3 * slope;
		}
		else if (activation_type == 3)
		{
		    float min = activation_params[0];
		    float max = activation_params[1];
		    if (sum0 < min) sum0 = min;
		    if (sum0 > max) sum0 = max;
		    if (sum1 < min) sum1 = min;
		    if (sum1 > max) sum1 = max;
		    if (sum2 < min) sum2 = min;
		    if (sum2 > max) sum2 = max;
		    if (sum3 < min) sum3 = min;
		    if (sum3 > max) sum3 = max;
		}

		top_blob[p] = sum0;
		top_blob[p+1] = sum1;
		top_blob[p+2] = sum2;
		top_blob[p+3] = sum3;
	    }
	    // num_output
	    #pragma omp parallel for num_threads(opt.num_threads)
	    for (int p=remain_num_output_start; p<num_output; p++)
	    {
		float sum = 0.f;

		if (bias_term)
		    sum = bias_data[p];

		const float* w = weight_data_ptr + size * channels * p;

	#if __ARM_NEON
		float32x4_t _sum = vdupq_n_f32(0.f);
		float32x4_t _sum2 = vdupq_n_f32(0.f);
	#endif // __ARM_NEON

		// channels
		for (int q=0; q<channels; q++)
		{
		    const float* m = bottom_blob.channel(q);

	#if __ARM_NEON
		    int nn = size >> 3;
		    int remain = size & 7;
	#else
		    int remain = size;
	#endif // __ARM_NEON

	#if __ARM_NEON
	#if __aarch64__
		    if (nn > 0)
		    {
		    asm volatile(
		        "0:                                   \n"
		        "prfm       pldl1keep, [%1, #256]     \n"
		        "ld1        {v0.4s, v1.4s}, [%1], #32 \n"
		        "prfm       pldl1keep, [%2, #256]     \n"
		        "ld1        {v2.4s, v3.4s}, [%2], #32 \n"
		        "fmla       %3.4s, v0.4s, v2.4s       \n"
		        "subs       %w0, %w0, #1              \n"
		        "fmla       %4.4s, v1.4s, v3.4s       \n"
		        "bne        0b                        \n"
		        : "=r"(nn),     // %0
		          "=r"(m),      // %1
		          "=r"(w),      // %2
		          "=w"(_sum),   // %3
		          "=w"(_sum2)   // %4
		        : "0"(nn),
		          "1"(m),
		          "2"(w),
		          "3"(_sum),
		          "4"(_sum2)
		        : "cc", "memory", "v0", "v1", "v2", "v3"
		    );
		    }
	#else
		    if (nn > 0)
		    {
		    asm volatile(
		        "0:                             \n"
		        "pld        [%1, #256]          \n"
		        "vld1.f32   {d0-d3}, [%1 :128]! \n"
		        "pld        [%2, #256]          \n"
		        "vld1.f32   {d4-d7}, [%2]!      \n"
		        "vmla.f32   %q3, q0, q2         \n"
		        "subs       %0, #1              \n"
		        "vmla.f32   %q4, q1, q3         \n"
		        "bne        0b                  \n"
		        : "=r"(nn),     // %0
		          "=r"(m),      // %1
		          "=r"(w),      // %2
		          "=w"(_sum),   // %3
		          "=w"(_sum2)   // %4
		        : "0"(nn),
		          "1"(m),
		          "2"(w),
		          "3"(_sum),
		          "4"(_sum2)
		        : "cc", "memory", "q0", "q1", "q2", "q3"
		    );
		    }
	#endif // __aarch64__
	#endif // __ARM_NEON
		    for (; remain>0; remain--)
		    {
		        sum += *m * *w;

		        m++;
		        w++;
		    }
		}

	#if __ARM_NEON
		_sum = vaddq_f32(_sum, _sum2);
	#if __aarch64__
		sum += vaddvq_f32(_sum);
	#else
		float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
		_sumss = vpadd_f32(_sumss, _sumss);
		sum += vget_lane_f32(_sumss, 0);
	#endif // __aarch64__
	#endif // __ARM_NEON

		if (activation_type == 1)
		{
		    sum = std::max(sum, 0.f);
		}
		else if (activation_type == 2)
		{
		    float slope = activation_params[0];
		    sum = sum > 0.f ? sum : sum * slope;
		}
		else if (activation_type == 3)
		{
		    float min = activation_params[0];
		    float max = activation_params[1];
		    if (sum < min)
		        sum = min;
		    if (sum > max)
		        sum = max;
		}
		else if (activation_type == 4)
		{
		    sum = 1.f / (1.f + exp(-sum));
		}

		top_blob[p] = sum;
	    }
	}//axis
  else if(axis==2)
  {	
    top_blob.create(num_output,h,channels,elemsize,opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* weight_data_ptr = weight_data;
//===================================sceond======================================
    double start = ncnn::get_current_time();
    #pragma omp parallel for num_threads(opt.num_threads)
    for(int q=0;q<channels;q++)
    {
        for(int i=0;i<h;i++)
        {
            for(int p =0;p<num_output;p++)
            {
                const float* m = bottom_blob.channel(q).row(i);//TODO
                float sum=0.f;
                const float* w0 =weight_data_ptr +w*p;

                if (bias_term)
                    sum = bias_data[p+num_output*i];
    #if __ARM_NEON
                float32x4_t _sum = vdupq_n_f32(0.f);
                float32x4_t _sum2 = vdupq_n_f32(0.f);
                int nn = w >>3;////30
                int remain = w & 7;////0
    #else
                int remain = w;
    #endif

    #if __ARM_NEON
    #if __aarch64__
                if (nn > 0)//30
                {
                asm volatile(
                    "0:                                   \n"
                    "prfm       pldl1keep, [%1, #256]     \n"
                    "ld1        {v0.4s, v1.4s}, [%1], #32 \n"
                    "prfm       pldl1keep, [%2, #256]     \n"
                    "ld1        {v2.4s, v3.4s}, [%2], #32 \n"
                    "fmla       %3.4s, v0.4s, v2.4s       \n"
                    "subs       %w0, %w0, #1              \n"
                    "fmla       %4.4s, v1.4s, v3.4s       \n"
                    "bne        0b                        \n"
                    : "=r"(nn),     // %0
                      "=r"(m),      // %1
                      "=r"(w0),      // %2
                      "=w"(_sum),   // %3
                      "=w"(_sum2)   // %4
                    : "0"(nn),
                      "1"(m),
                      "2"(w0),
                      "3"(_sum),
                      "4"(_sum2)
                    : "cc", "memory", "v0", "v1", "v2", "v3"
                );
                }
    #else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1 :128]! \n"
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d4-d7}, [%2]!      \n"
                    "vmla.f32   %q3, q0, q2         \n"
                    "subs       %0, #1              \n"
                    "vmla.f32   %q4, q1, q3         \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(m),      // %1
                      "=r"(w0),      // %2
                      "=w"(_sum),   // %3
                      "=w"(_sum2)   // %4
                    : "0"(nn),
                      "1"(m),
                      "2"(w0),
                      "3"(_sum),
                      "4"(_sum2)
                    : "cc", "memory", "q0", "q1", "q2", "q3"
                );
                }
    #endif // __aarch64__
    #endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    sum += *m * *w0;

                    m++;
                    w0++;
                }
    #if __ARM_NEON
            _sum = vaddq_f32(_sum, _sum2);
    #if __aarch64__
            sum += vaddvq_f32(_sum);
    #else
            float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
            _sumss = vpadd_f32(_sumss, _sumss);
            sum += vget_lane_f32(_sumss, 0);
    #endif // __aarch64__
    #endif // __ARM_NEON
             top_blob.channel(q)[p+num_output*i]=sum;
            }
        }
    }
//================================first=========================================
    /*int nn_num_output = num_output >> 2;////3
    int remain_num_output_start = nn_num_output << 2;
    double start = ncnn::get_current_time();
    #pragma omp parallel for num_threads(opt.num_threads)
    for(int q=0;q<channels;q++)
    {
        for(int i=0;i<h;i++)
        {
            for(int pp =0;pp<nn_num_output;pp++)/// 0 1 2
            {
                const float* m = bottom_blob.channel(q).row(i);

                int p=pp*4;///4
                float sum0=0.f;
                float sum1=0.f;
                float sum2=0.f;
                float sum3=0.f;

                const float* w0 =weight_data_ptr +w*p;///240*4
                const float* w1 =weight_data_ptr +w*(p+1);///240*5
                const float* w2 =weight_data_ptr +w*(p+2);////240*6
                const float* w3 =weight_data_ptr +w*(p+3);////240*7

                if(bias_term)
                {
                    sum0 = bias_data[p+num_output*i];///  0
                    sum1 = bias_data[p+1+num_output*i];///  1
                    sum2 = bias_data[p+2+num_output*i];//// 2
                    sum3 = bias_data[p+3+num_output*i];////  3
                }
    #if __ARM_NEON
                float32x4_t _sum0 = vdupq_n_f32(0.f);
                float32x4_t _sum1 = vdupq_n_f32(0.f);
                float32x4_t _sum2 = vdupq_n_f32(0.f);
                float32x4_t _sum3 = vdupq_n_f32(0.f);

                int nn = w >> 2;//60
                int remain = w & 3;///0
    #else
                int remain = w;
    #endif // __ARM_NEON


    #if __ARM_NEON
                for (; nn>0; nn--)//60
                {
                    float32x4_t _m = vld1q_f32(m);

                    float32x4_t _w0 = vld1q_f32(w0);
                    _sum0 = vmlaq_f32(_sum0, _m, _w0);

                    float32x4_t _w1 = vld1q_f32(w1);
                    _sum1 = vmlaq_f32(_sum1, _m, _w1);

                    float32x4_t _w2 = vld1q_f32(w2);
                    _sum2 = vmlaq_f32(_sum2, _m, _w2);

                    float32x4_t _w3 = vld1q_f32(w3);
                    _sum3 = vmlaq_f32(_sum3, _m, _w3);
                    m += 4;
                    w0 += 4;
                    w1 += 4;
                    w2 += 4;
                    w3 += 4;
                }
                float32x2_t _sum0ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                float32x2_t _sum1ss = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                float32x2_t _sum2ss = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
                float32x2_t _sum3ss = vadd_f32(vget_low_f32(_sum3), vget_high_f32(_sum3));

                float32x2_t _sum01ss = vpadd_f32(_sum0ss, _sum1ss);
                float32x2_t _sum23ss = vpadd_f32(_sum2ss, _sum3ss);

                sum0 += vget_lane_f32(_sum01ss, 0);
                sum1 += vget_lane_f32(_sum01ss, 1);
                sum2 += vget_lane_f32(_sum23ss, 0);
                sum3 += vget_lane_f32(_sum23ss, 1);
    #endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    sum0 += *m * *w0;
                    sum1 += *m * *w1;
                    sum2 += *m * *w2;
                    sum3 += *m * *w3;

                    m++;
                    w0++;
                    w1++;
                    w2++;
                    w3++;
                }
                top_blob.channel(q)[p+num_output*i]=sum0;
                top_blob.channel(q)[p+1+num_output*i]=sum1;
                top_blob.channel(q)[p+2+num_output*i]=sum2;
                top_blob.channel(q)[p+3+num_output*i]=sum3;
            }
        }
    }
    // remain_num_output
    if(remain_num_output_start!=num_output)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for(int q=0;q<channels;q++)
        {
            for(int i=0;i<h;i++)
            {
                for(int p =remain_num_output_start;p<num_output;p++)
                {
                    const float* m = bottom_blob.channel(q).row(i);
                    float sum=0.f;
                    const float* w0 =weight_data_ptr +w*p;

                    if (bias_term)
                        sum = bias_data[p+num_output*i];
            #if __ARM_NEON
                    float32x4_t _sum = vdupq_n_f32(0.f);
                    float32x4_t _sum2 = vdupq_n_f32(0.f);
                    int nn = w >>3;////30
                    int remain = w & 7;///0
            #else
                    int remain = w;
            #endif

            #if __ARM_NEON
            #if __aarch64__
                    if (nn > 0)//30
                    {
                    asm volatile(
                        "0:                                   \n"
                        "prfm       pldl1keep, [%1, #256]     \n"
                        "ld1        {v0.4s, v1.4s}, [%1], #32 \n"
                        "prfm       pldl1keep, [%2, #256]     \n"
                        "ld1        {v2.4s, v3.4s}, [%2], #32 \n"
                        "fmla       %3.4s, v0.4s, v2.4s       \n"
                        "subs       %w0, %w0, #1              \n"
                        "fmla       %4.4s, v1.4s, v3.4s       \n"
                        "bne        0b                        \n"
                        : "=r"(nn),     // %0
                          "=r"(m),      // %1
                          "=r"(w0),      // %2
                          "=w"(_sum),   // %3
                          "=w"(_sum2)   // %4
                        : "0"(nn),
                          "1"(m),
                          "2"(w0),
                          "3"(_sum),
                          "4"(_sum2)
                        : "cc", "memory", "v0", "v1", "v2", "v3"
                    );
                    }
            #else
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n"
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2]!      \n"
                        "vmla.f32   %q3, q0, q2         \n"
                        "subs       %0, #1              \n"
                        "vmla.f32   %q4, q1, q3         \n"
                        "bne        0b                  \n"
                        : "=r"(nn),     // %0
                          "=r"(m),      // %1
                          "=r"(w0),      // %2
                          "=w"(_sum),   // %3
                          "=w"(_sum2)   // %4
                        : "0"(nn),
                          "1"(m),
                          "2"(w0),
                          "3"(_sum),
                          "4"(_sum2)
                        : "cc", "memory", "q0", "q1", "q2", "q3"
                    );
                    }
            #endif // __aarch64__
            #endif // __ARM_NEON
                    for (; remain>0; remain--)
                    {
                        sum += *m * *w0;

                        m++;
                        w0++;
                    }
            #if __ARM_NEON
                _sum = vaddq_f32(_sum, _sum2);
            #if __aarch64__
                sum += vaddvq_f32(_sum);
            #else
                float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                _sumss = vpadd_f32(_sumss, _sumss);
                sum += vget_lane_f32(_sumss, 0);
            #endif // __aarch64__
            #endif // __ARM_NEON
                 top_blob.channel(q)[p+num_output*i]=sum;
                }
            }
        }
    }*/
 //=============================last==========================================
     /* double start = ncnn::get_current_time();
      #pragma omp parallel for num_threads(opt.num_threads)
      for(int q=0;q<channels;q++)///8
      {
        const float* m = bottom_blob.channel(q);
        for(int i=0;i<h;i++)////1
        {
            for(int j=0;j<num_output;j++)///12
            {
                float sum=0.f;
                const float* w0 =weight_data_ptr +w*j;
                if(bias_term)
                {
                    sum = bias_data[j+num_output*i];
                }
                for(int p=0;p<w;p++)////240
                {
                    sum +=m[p+w*i] * w0[p];
                }
                top_blob.channel(q)[j+num_output*i]=sum;
            }
        }
      }*/
    double end = ncnn::get_current_time();
    double time = end - start;
    std::cout<<"neon_time: "<<time<<std::endl;
 }//axis2
else
{
    return -1;
}
    return 0;
}

} // namespace ncnn
