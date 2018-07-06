#define _CRT_SECURE_NO_WARNINGS

#include "Header.h"
#include <stdio.h>
//#include <tchar.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
double activation_function(double value, int function);
double activation_d_function(double value, int function);
double cost_d_function(double actual_value, double output_value);
double cost_function(double actual_value, double output_value);
void cal_forward(ANN_LAYER *input_layer, ANN_LAYER *output_layer, ANN_WEIGHT *weight)
{
	//printf("%d", train->net_setting.input_nodes_num);

	//for (i = 0; i < train->net_setting.input_nodes_num; i++) {
		//printf("%d", train->net_setting.input_nodes_num);
		//printf("%lf,", input_layer->transfer_sum[0]);
	//}
	//printf("\n");
	int i=0, j, k;
	double tmp = 0;
	for (j = 0; j<weight->behind_layer; j++) {
		tmp = 0;
		for (k = 0; k<weight->front_layer; k++) {
			tmp += input_layer->transfer_sum[k] * weight->weight_value[k][j];
			//printf("%d,%f=%f*%f\n",i,tmp, train->layer[i].nodes_value[k], train->weight[i-1].weight_value[k][j]);
		}
		//system("pause");
		output_layer->transfer_sum[j] = activation_function(tmp + output_layer->bias[j], input_layer->activation_function);
	}
}

void cal_ann_forward_output(NNET *train, ANN_LAYER *input_layer)
{
	int j;
	double tmp = 0;
	double tmp2 = 0;

	for (j = 0; j<train->net_setting.output_nodes_num; j++) {
		tmp += exp(input_layer->transfer_sum[j]);
	}
	for (j = 0; j<train->net_setting.output_nodes_num; j++) {
		tmp2 = exp(input_layer->transfer_sum[j]);
		train->softmax_layer.softmax_value[j] = (double)(tmp2 / tmp);
	}

}

void cal_ann_layer_forward(ANN_LAYER *input_layer, ANN_LAYER *output_layer, ANN_WEIGHT *weight,int use_drop)
{

	
	//printf("%d", train->net_setting.input_nodes_num);
	int i = 0, j, k;
	double tmp = 0;
	//for (i = 0; i < train->net_setting.input_nodes_num; i++) {
	//printf("%d", train->net_setting.input_nodes_num);
	//printf("%lf,", input_layer->transfer_sum[0]);
	//}
	//printf("\n");
	if (use_drop == 0) {
		for (j = 0; j<weight->behind_layer; j++) {
			tmp = 0;
			for (k = 0; k<weight->front_layer; k++) {
				tmp += input_layer->transfer_sum[k] * weight->weight_value[k][j];
				//printf("%d,%f=%f*%f\n",i,tmp, train->layer[i].nodes_value[k], train->weight[i-1].weight_value[k][j]);
			}
			//system("pause");
			output_layer->transfer_sum[j] = activation_function(tmp + output_layer->bias[j], input_layer->activation_function);
		}
	}
	else if (use_drop == 1) {
		/*for (i = 0; i < output_layer->nodes_num; i++) {
			rand_num = (double)((rand() % 10)/10);
			if (output_layer->drop_prob <= rand_num) {
				output_layer->is_drop[i] = 1;
			}
			else {
				output_layer->is_drop[i] = 0;
			}
		}*/

		for (j = 0; j<weight->behind_layer; j++) {
			tmp = 0;
			for (k = 0; k<weight->front_layer; k++) {
				tmp += (double)input_layer->transfer_sum[k] * weight->weight_value[k][j]*(1.0/(1.0- input_layer->drop_prob));
				//printf("%d,%f=%f*%f\n",i,tmp, train->layer[i].nodes_value[k], train->weight[i-1].weight_value[k][j]);
			}
			//system("pause");
			output_layer->transfer_sum[j] = activation_function(tmp + output_layer->bias[j], input_layer->activation_function)* output_layer->is_drop[j];
		}
	}
	
	
}

void determine_is_drop(NNET &b)
{
	srand((int)(time(NULL)));
	double rand_num;

	int i, j;
	for (j = 0; j < b.net_setting.layers_num; j++) {
		for (i = 0; i < b.layer[j].nodes_num; i++) {
			rand_num = (double)((rand() % 10) / 10);
			if (b.layer[j].drop_prob <= rand_num) {
				b.layer[j].is_drop[i] = 1;
			}
			else {
				b.layer[j].is_drop[i] = 0;
			}
		}
	}

}

void cnn_to_ann(NNET &b)
{
	int i, j, k;
	int counter = 0;
	for (i = 0; i < b.conv_output[b.net_setting.convolutional_layer_num - 1].layer_num; i++) {
		for (j = 0; j < b.conv_output[b.net_setting.convolutional_layer_num - 1].row; j++) {
			for (k = 0; k < b.conv_output[b.net_setting.convolutional_layer_num - 1].column; k++) {
				b.layer[0].transfer_sum[counter] = b.conv_output[b.net_setting.convolutional_layer_num - 1].conv_out_layer[i].matrix[j][k];
				counter++;
			}
		}
	}

}

void ann_to_cnn(NNET &b)
{
	int i, j, k;
	int counter = 0;
	for (i = 0; i < b.conv_output[b.net_setting.convolutional_layer_num - 1].layer_num; i++) {
		for (j = 0; j < b.conv_output[b.net_setting.convolutional_layer_num - 1].row; j++) {
			for (k = 0; k < b.conv_output[b.net_setting.convolutional_layer_num - 1].column; k++) {
				b.conv_output[b.net_setting.convolutional_layer_num - 1].conv_out_layer[i].matrix_deviation[j][k] = b.layer[0].deviation_value[counter];
				counter++;
			}
		}
	}
}