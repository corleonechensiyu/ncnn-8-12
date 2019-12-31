#ifndef LAYER_CTCDECODER_H
#define LAYER_CTCDECODER_H

#include "layer.h"
namespace ncnn{
	
class CTCGreedyDecoder : public Layer
{
	public:
		CTCGreedyDecoder();
		virtual int load_param(const ParamDict& pd);
		//virtual int forward_inplace(Mat& bottom_top_blob,const Option& opt) const;
		virtual int forward(const Mat& bottom_blob,Mat& top_blob,const Option& opt) const;
	public:
		int blank;
};


}
#endif 
