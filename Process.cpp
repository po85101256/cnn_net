#define _CRT_SECURE_NO_WARNINGS

#include<stdio.h>
#include "Header.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
void train_net(NNET &b, DATA_SET &d, int first, int last);

void cal_cnn_layer_forward(CNN_OUTPUT *front_output, CNN_MASK *conv_mask, CNN_OUTPUT *output);
void cal_cnn_layer_sensitive_map(CNN_OUTPUT *front_output, CNN_MASK *conv_mask, CNN_OUTPUT *output);
void cal_cnn_mask_update(NNET &b, CNN_OUTPUT *front_output, CNN_MASK *conv_mask, CNN_OUTPUT *output);
void cal_cnn_mask_update_first(NNET &b, DATA_SET &d, int n, CNN_MASK *conv_mask, CNN_OUTPUT *output);

void cal_ann_layer_forward(ANN_LAYER *input_layer, ANN_LAYER *output_layer, ANN_WEIGHT *weight, int use_drop);
void cal_cnn_mask_bias_update(NNET &b, CNN_OUTPUT *output, CNN_MASK *conv_mask);
void cal_ann_forward_output(NNET *train, ANN_LAYER *input_layer);
void cal_forward(ANN_LAYER *input_layer, ANN_LAYER *output_layer, ANN_WEIGHT *weight);
void ann_cal_deviation_value(NNET &b, ANN_SOFTMAX &softmax_layer, ANN_LAYER *output_layer, DATA_SET &d, int n);
void ann_cal_deviation_backward(NNET &b, ANN_LAYER *layer, ANN_LAYER *front_layer, ANN_WEIGHT *weight, int use_drop);
void ann_cal_weight_backward(NNET &b, ANN_LAYER *layer, ANN_LAYER *front_layer, ANN_WEIGHT *weight, int use_drop);

void set_data_to_ann(NNET &b, DATA_SET &d, int n);
//void set_data_to_cnn(NNET &b, DATA_SET &d, int n);
void ann_cal_forward(NNET &b, DATA_SET &d, int n);
void ann_cal_backward(NNET &b, DATA_SET &d, int n);
void cnn_cal_forward(NNET &b, DATA_SET &d, int n);
void cnn_cal_backward(NNET &b, DATA_SET &d, int n);

void printf_cnn(NNET &b);
void printf_ann(NNET &b);
void printf_ann_b(NNET &b);
void printf_cnn_b(NNET &b);

void cnn_to_ann(NNET &b);
void ann_to_cnn(NNET &b);
void determine_is_drop(NNET &b);
double cost_function(double actual_value, double output_value);
void cal_data_cnn(NNET &b, DATA_SET &d, int n);

void train_net(NNET &b, DATA_SET &d, int first, int last)
{
	int i;
	//double tmp2 = 0;
	//for (i = 0; i<1; i++) {
	for (i = first; i<last; i++) {
		//printf_cnn(b);
		cnn_cal_forward(b, d, i);
		///

		cnn_to_ann(b);
		ann_cal_forward(b, d, i);
		//	printf_cnn(b);
		//printf_ann(b);
		ann_cal_backward(b, d, i);
		//printf_ann_b(b);

		ann_to_cnn(b);
		cnn_cal_backward(b, d, i);
	}
}

void cnn_cal_forward(NNET &b, DATA_SET &d, int n)
{
	int i;
	cal_data_cnn(b, d, n);
	for (i = 1; i < b.net_setting.convolutional_layer_num; i++)
	{
		cal_cnn_layer_forward(&b.conv_output[i - 1], &b.conv_layer[i], &b.conv_output[i]);
	}
}

void cnn_cal_backward(NNET &b, DATA_SET &d, int n)
{
	int i;
	for (i = b.net_setting.convolutional_layer_num - 1; i >= 1; i--) {
		cal_cnn_layer_sensitive_map(&b.conv_output[i - 1], &b.conv_layer[i], &b.conv_output[i]);

	}
	//printf_cnn_b(b);
	for (i = b.net_setting.convolutional_layer_num - 1; i >= 1; i--) {
		cal_cnn_mask_update(b, &b.conv_output[i - 1], &b.conv_layer[i], &b.conv_output[i]);

	}
	cal_cnn_mask_update_first(b, d, n, &b.conv_layer[0], &b.conv_output[0]);
	for (i = 0; i<b.net_setting.convolutional_layer_num; i++) {
		cal_cnn_mask_bias_update(b, &b.conv_output[i], &b.conv_layer[i]);

	}

}


void ann_cal_forward(NNET &b, DATA_SET &d, int n)
{
	int i;
	//set_data_to_ann(b,d,n);

	if (b.net_setting.is_drop == 1) {///check is use dropout
		determine_is_drop(b);
	}

	for (i = 0; i < b.net_setting.layers_num - 1; i++)
	{
		cal_ann_layer_forward(&b.layer[i], &b.layer[i + 1], &b.weight[i], b.net_setting.is_drop);
	}
	cal_ann_forward_output(&b, &b.layer[b.net_setting.layers_num - 1]);
}

void ann_cal_backward(NNET &b, DATA_SET &d, int n)
{
	int i;
	ann_cal_deviation_value(b, b.softmax_layer, &b.layer[b.net_setting.layers_num - 1], d, n);

	for (i = b.net_setting.layers_num - 1; i >= 1; i--)
	{
		ann_cal_deviation_backward(b, &b.layer[i], &b.layer[i - 1], &b.weight[i - 1], b.net_setting.is_drop);

	}
	for (i = b.net_setting.layers_num - 1; i >= 1; i--)
	{
		ann_cal_weight_backward(b, &b.layer[i], &b.layer[i - 1], &b.weight[i - 1], b.net_setting.is_drop);

	}
}

void set_data_to_ann(NNET &b, DATA_SET &d, int n)
{
	int i;
	for (i = 0; i < b.net_setting.input_nodes_num; i++) {
		b.layer[0].transfer_sum[i] = d.data_layer[n].input_data[i];
	}
}

/*void set_data_to_cnn(NNET &b, DATA_SET &d, int n)
{
	int i;

}*/