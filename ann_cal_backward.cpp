#define _CRT_SECURE_NO_WARNINGS

#include "Header.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
double activation_d_function(double value, int function);
double cost_d_function(double actual_value, double output_value);
void ann_cal_deviation_value(NNET &b, ANN_SOFTMAX &softmax_layer, ANN_LAYER *output_layer, DATA_SET &d, int n)
{
	int i;
	
	double tmp2 = 0;
	
	for (i = 0; i < output_layer->nodes_num; i++) {
		//printf("%d", i);
		output_layer->deviation_value[i] = activation_d_function(output_layer->transfer_sum[i], output_layer->activation_function)
			*cost_d_function(d.data_layer[n].output_data[i], softmax_layer.softmax_value[i]);
		output_layer->bias[i] -= b.net_setting.learning_rate * (output_layer->deviation_value[i]);
		//printf("(%f,%f,%f)\n",output_layer->transfer_sum[i],output_layer->nodes_value[i],output_layer->deviation_value[i]);
	}

}

void ann_cal_deviation_backward(NNET &b, ANN_LAYER *layer, ANN_LAYER *front_layer,ANN_WEIGHT *weight,  int use_drop)
{
	int  j, k;
	double tmp=0;
	if (use_drop == 0) {
		for (j = 0; j<front_layer->nodes_num; j++) {
			tmp = 0;
			for (k = 0; k<weight->behind_layer; k++) {
				//printf("(%d,%d)=(%d,%d)*(%d,%d)\n",i,j,i+1,k,j,k);
				tmp += activation_d_function(front_layer->transfer_sum[j], front_layer->activation_function)*layer->deviation_value[k] * weight->weight_value[j][k];
			}
			front_layer->deviation_value[j] = (double)tmp;
			front_layer->bias[j] -= b.net_setting.learning_rate * front_layer->deviation_value[j];

		}
	}
	else if (use_drop == 1) {
		for (j = 0; j<front_layer->nodes_num; j++) {
			tmp = 0;
			for (k = 0; k<weight->behind_layer; k++) {
				//printf("(%d,%d)=(%d,%d)*(%d,%d)\n",i,j,i+1,k,j,k);
				tmp += activation_d_function(front_layer->transfer_sum[j], front_layer->activation_function)*layer->deviation_value[k] * weight->weight_value[j][k];
			}
			front_layer->deviation_value[j] = (double)(tmp*front_layer->is_drop[j]*(double)(1/(1-front_layer->drop_prob)));
			front_layer->bias[j] -= b.net_setting.learning_rate * front_layer->deviation_value[j];

		}
	}
	
}

void ann_cal_weight_backward(NNET &b, ANN_LAYER *layer, ANN_LAYER *front_layer, ANN_WEIGHT *weight, int use_drop)
{
	int j, k;
	if (use_drop == 0) {
		for (j = 0; j<weight->front_layer; j++) {
			for (k = 0; k<weight->behind_layer; k++) {
				weight->weight_value[j][k] -= b.net_setting.learning_rate*layer->deviation_value[k] * front_layer->transfer_sum[j];
			}
		}
	}
	else if (use_drop == 1) {
		for (j = 0; j<weight->front_layer; j++) {
			for (k = 0; k<weight->behind_layer; k++) {
				weight->weight_value[j][k] -= b.net_setting.learning_rate*layer->deviation_value[k] * front_layer->transfer_sum[j]*front_layer->is_drop[j];
			}
		}
	}
	
}