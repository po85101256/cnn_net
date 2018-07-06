#define _CRT_SECURE_NO_WARNINGS

#include "Header.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
double activation_function(double value, int function);
double activation_d_function(double value, int function);
double cost_d_function(double actual_value, double output_value);
double cost_function(double actual_value, double output_value);

void cal_data_cnn(NNET &b, DATA_SET &d, int n)
{
	int i, j, k, l, m;
	double tmp2=0;
	double tmp=0;
	if (b.conv_layer[0].layer_type == 0) {
		for (i = 0; i < b.conv_layer[0].mask_num; i++) {
			for (j = 0; j < b.conv_output[0].row; j++) {
				for (k = 0; k < b.conv_output[0].column; k++) {
					for (l = 0; l < b.conv_layer[0].mask_row; l++) {
						for (m = 0; m < b.conv_layer[0].mask_column; m++) {
							tmp2 += b.conv_layer[0].mask[i].weight[l][m] * d.data_layer[n].input_data[(l*b.net_setting.input_matrix_column) + m
								+ (j*b.conv_layer[0].down_step*b.net_setting.input_matrix_column) + k * b.conv_layer[0].left_step];
						}
					}
					b.conv_output[0].conv_out_layer[i].matrix[j][k] = activation_function(tmp2 + b.conv_layer[0].mask_bias[i], b.conv_layer[0].activation_function_type);
					tmp2 = 0;
				}
			}
		}
	}
	/*else if (b.conv_layer[0].layer_type == 1) {
		for (k = 0; k < b.conv_output[0].layer_num; k++) {
			for (i = 0; i < b.conv_output[0].row; i = i + b.conv_layer[0].mask_row) {
				for (j = 0; j < b.conv_output[0].column; j = j + b.conv_layer[0].mask_column) {
					tmp = train->convolutional_output_layer[c - 1].con_out_layer[k].matrix[i][j];
					for (a = 0; a < train->convolutional_layer[c].filter_row; a++) {
						for (b = 0; b < train->convolutional_layer[c].filter_column; b++) {
							if (train->convolutional_output_layer[c - 1].con_out_layer[k].matrix[i + a][j + b] >= tmp) {
								tmp = train->convolutional_output_layer[c - 1].con_out_layer[k].matrix[i + a][j + b];
							}
						}
					}

					train->convolutional_output_layer[c].con_out_layer[k].matrix[(int)(i / train->convolutional_layer[c].filter_row)][(int)(j / train->convolutional_layer[c].filter_column)] = tmp;
				}
			}
		}
	}*/
	
}


void cal_cnn_layer_forward(CNN_OUTPUT *front_output, CNN_MASK *conv_mask, CNN_OUTPUT *output)
{
	int i, j, k, l, m, n;
	double tmp2 = 0;
	double tmp = 0;
	if (conv_mask->layer_type == 0) {
		for (i = 0; i < conv_mask->mask_num; i++) {
			for (j = 0; j < output->row; j++) {
				for (k = 0; k < output->column; k++) {
					for (l = 0; l < front_output->layer_num; l++) {
						for (m = 0; m < conv_mask->mask_row; m++) {
							for (n = 0; n < conv_mask->mask_column; n++) {
								tmp2 += conv_mask->mask[i].weight[m][n] * front_output->conv_out_layer[l].matrix[j + m][k + n];
							}
						}
					}
					output->conv_out_layer[i].matrix[j][k] = activation_function(tmp2 + conv_mask->mask_bias[i], conv_mask->activation_function_type);
					tmp2 = 0;
				}
			}
		}
	}
	else if (conv_mask->layer_type == 1) {
		for (i = 0; i < front_output->layer_num; i++) {
			for (j = 0; j < front_output->row; j = j + conv_mask->down_step) {
				for (k = 0; k < front_output->column; k = k + conv_mask->left_step) {
					tmp = front_output->conv_out_layer[i].matrix[j][k];
					for (l = 0; l < conv_mask->mask_row; l++) {
						for (m = 0; m < conv_mask->mask_column; m++) {
							if (front_output->conv_out_layer[i].matrix[j + l][k + m] >= tmp) {
								tmp = front_output->conv_out_layer[i].matrix[j + l][k + m];
							}
						}
					}
					output->conv_out_layer[i].matrix[(int)(j / conv_mask->mask_row)][(int)(k / conv_mask->mask_column)] = tmp;
				}
			}
		}
	}

}
