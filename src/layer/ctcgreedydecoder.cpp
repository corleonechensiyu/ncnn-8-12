#include "ctcgreedydecoder.h"
#include <algorithm>
#include <math.h>
#include <vector>
#include <iostream>

using namespace std;
namespace ncnn{

DEFINE_LAYER_CREATOR(CTCGreedyDecoder)

CTCGreedyDecoder::CTCGreedyDecoder()
{
	one_blob_only = true;
	support_inplace = false;
}

int CTCGreedyDecoder::load_param(const ParamDict& pd)
{
	blank = pd.get(0,0);
	return 0;
}

int CTCGreedyDecoder::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
	int T =bottom_blob.c;
	int N =bottom_blob.h;
	int C =bottom_blob.w;
	size_t elemsize = bottom_blob.elemsize;
	vector<int> Sequence;
	Sequence.clear();
	top_blob.create(T, elemsize, opt.blob_allocator);
	if(top_blob.empty())
		return -100;


	#pragma omp parallel for num_threads(opt.num_threads)
	for(int n=0;n<N;n++)
	{
		int prev_class_idx=-1;
		for(int t=0;t<T;++t)
		{
			int max_class_idx=0;
			const float* m = bottom_blob.channel(t);
			float max_prob = m[0];
			++m;
			for(int c=1;c<C;++c,++m)
			{
				if(*m>max_prob)
				{
					max_class_idx = c;
					max_prob = *m;
				}
			}
			if(max_class_idx != blank && !(max_class_idx == prev_class_idx))
			{
				Sequence.push_back(max_class_idx);
			}
			prev_class_idx = max_class_idx;
		}
	}
 
	int size_ocr=Sequence.size();
	for(int i=0;i< size_ocr;i++)
	{
		top_blob[i]=Sequence[i];
	}





	return 0;
}


}
