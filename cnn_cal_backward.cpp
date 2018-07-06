#define _CRT_SECURE_NO_WARNINGS

#include "Header.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//void zero_padding(double **map, CNN_OUTPUT *output);
double activation_d_function(double value,  int function);
void do_inverse_mask(CNN_MASK *mask, int num);
void do_zero_padding(double **tmp_padding, CNN_OUTPUT *output, int num, int row, int column);

void do_inverse_mask(CNN_MASK *mask,int num)
{
	int j, k;
	//for (i = 0; i < mask->filter_num; i++) {
		for (j = 0; j < mask->mask[num].row; j++) {
			for (k = 0; k < mask->mask[num].column; k++) {
				mask->inverse_mask[j][k] = mask->mask[num].weight[(mask->mask[num].row - 1) - j][(mask->mask[num].column - 1) - k];
				//printf("%lf", train->convolutional_layer[i].inverse_filter[a][b]);
			}
			//	printf("\n");
		}

		/*printf("inverse_mask:\n");
		for (j = 0; j < mask->mask[num].row; j++) {
			for (k = 0; k < mask->mask[num].column; k++) {
				//mask->inverse_mask[j][k] = mask->mask[num].weight[(mask->mask[num].row - 1) - j][(mask->mask[num].column - 1) - k];
				printf("%lf,", mask->inverse_mask[j][k]);
			}
				printf("\n");
		}*/
	//}
	
}

void do_zero_padding(double **tmp_padding,CNN_OUTPUT *output, int num,int row ,int column)
{
	int i, j;
		for (i = 0; i < output->row + ((row - 1) * 2); i++) {
			for (j = 0; j < output->column + ((column - 1) * 2); j++) {
				if (i >= ((row - 1) ) && i < (output->row+ row-1)) {
					if (j >= ((column - 1) ) && j < (output->column+ column-1)) {
						tmp_padding[i][j] = output->conv_out_layer[num].matrix_deviation[i - (row - 1)][j - (column - 1)];
					}
					else {
						tmp_padding[i][j] = 0;
					}
				}
				else {
					tmp_padding[i][j] = 0;
				}
			}
		}
		/*printf("zero_padding:\n");
		for (i = 0; i < output->row + ((row - 1) * 2); i++) {
			for (j = 0; j < output->column + ((column - 1) * 2); j++) {
				printf("%lf,", tmp_padding[i][j]);
			}
			printf("\n");
		}*/
	
}

void cal_cnn_layer_sensitive_map(CNN_OUTPUT *front_output, CNN_MASK *conv_mask, CNN_OUTPUT *output)
{
	int i, j, k,l,m;
	double tmp;
	tmp = 0;
	int pool_i;
	int pool_j;

	double **t_sensitive_map = NULL;
	t_sensitive_map = new double *[front_output->row];
	for (i = 0; i < front_output->row; i++) {
		t_sensitive_map[i] = new double[front_output->column];
	}
	for (i = 0; i < front_output->row; i++) {
		for (j = 0; j < front_output->column; j++) {
			t_sensitive_map[i][j] = 0;
		}
	}
	if (conv_mask->layer_type == 0) {
		///
		double **tmp_padding = NULL;
		tmp_padding = new double *[output->row + ((conv_mask->mask_row - 1) * 2)];
		for (j = 0; j < output->row + ((conv_mask->mask_row - 1) * 2); j++) {
			tmp_padding[j] = new double[output->column + ((conv_mask->mask_column - 1) * 2)];
		}
		///
		for (i = 0; i < output->layer_num; i++) {
			do_inverse_mask(conv_mask, i);
			do_zero_padding(tmp_padding, output, i, conv_mask->mask_row, conv_mask->mask_column);
			for (j = 0; j < front_output->row; j++) {
				for (k = 0; k < front_output->column; k++) {
					for (l = 0; l < conv_mask->mask_row; l++) {
						for (m = 0; m < conv_mask->mask_column; m++) {
							t_sensitive_map[j][k] += conv_mask->inverse_mask[l][m] * tmp_padding[j + l][k + m];
						}
					}
				}
			}
		}
		///
		for (i = 0; i < front_output->layer_num; i++) {
			for (j = 0; j < front_output->row; j++) {
				for (k = 0; k < front_output->column; k++) {
					front_output->conv_out_layer[i].matrix_deviation[j][k] = activation_d_function(front_output->conv_out_layer[i].matrix[j][k], conv_mask->activation_function_type) * t_sensitive_map[j][k];
				}
			}
		}
		///


		for (j = 0; j < output->row + ((conv_mask->mask_row - 1) * 2); j++) {
			delete[]tmp_padding[j];
		}
		delete[]tmp_padding;
	}
	else if (conv_mask->layer_type == 1) {

		for (i = 0; i < front_output->layer_num; i++) {
			for (j = 0; j < front_output->row; j = j + conv_mask->left_step) {
				for (k = 0; k < front_output->column; k = k + conv_mask->down_step) {
					tmp = front_output->conv_out_layer[i].matrix[j][k];
					for (l = 0; l < conv_mask->mask_row; l++) {
						for (m = 0; m < conv_mask->mask_column; m++) {
							if (front_output->conv_out_layer[i].matrix[j + l][k + m] >= tmp) {
								tmp = front_output->conv_out_layer[i].matrix[j + l][k + m];
								pool_i = j + l;
								pool_j = k + m;
							}
						}
					}
					for(l = 0; l < conv_mask->mask_row; l++) {
						for (m = 0; m < conv_mask->mask_column; m++) {
							if (((j + l) == pool_i) && ((k + m) == pool_j)) {
								t_sensitive_map[pool_i][pool_j] += output->conv_out_layer[i].matrix_deviation[(int)(j / conv_mask->left_step)][(int)(k / conv_mask->down_step)];
							}
							else {
								//front_output->conv_out_layer[i].matrix_deviation[j + l][k + m] = 0;
							}
						}
					}
				}
			}
			for (j = 0; j < front_output->row; j++) {
				for (k = 0; k < front_output->column; k++) {
					front_output->conv_out_layer[i].matrix_deviation[j][k] = t_sensitive_map[j][k];
					t_sensitive_map[j][k] = 0;
				}
			}

		}
	}
	
	for (i = 0; i < front_output->row; i++) {
		delete[]t_sensitive_map[i];
		t_sensitive_map[i] = NULL;
	}delete[]t_sensitive_map;
	t_sensitive_map = NULL;
	///
}

void cal_cnn_mask_update(NNET &b,CNN_OUTPUT *front_output, CNN_MASK *conv_mask, CNN_OUTPUT *output)
{
	int i, j, k,l,m,n;
	double tmp = 0;
	for (i = 0; i < conv_mask->mask_num; i++) {
		for (j = 0; j < conv_mask->mask_row; j++) {
			for (k = 0; k < conv_mask->mask_column; k++) {
				for (l = 0; l < front_output->layer_num; l++) {
					for (m = 0; m < output->row; m++) {
						for (n = 0; n < output->column; n++) {
							tmp += output->conv_out_layer[i].matrix_deviation[m][n] * front_output->conv_out_layer[l].matrix[j + m][k + n];
							//	tmp += train->convolutional_layer[i].filter[k].weight[k][j] * train->convolutional_output_layer[i - 1].con_out_layer[l].matrix[a][b];
						}
					}
				}
				conv_mask->mask[k].weight[j][k] -= b.net_setting.learning_rate * tmp;
				tmp = 0;
			}
		}
	}
	
}

void cal_cnn_mask_update_first(NNET &b, DATA_SET &d, int n, CNN_MASK *conv_mask, CNN_OUTPUT *output)
{
	int i, j, k, l, m;
	double tmp = 0;

	for (i = 0; i < conv_mask->mask_num; i++) {
		for (j = 0; j < conv_mask->mask_row; j++) {
			for (k = 0; k < conv_mask->mask_column; k++) {
				for (l = 0; l < output->row; l++) {
					for (m = 0; m < output->column; m++) {
						tmp += output->conv_out_layer[i].matrix_deviation[l][m] * d.data_layer[n].input_data[(l*b.net_setting.input_matrix_column) + m];
						//	tmp += train->convolutional_layer[i].filter[k].weight[k][j] * train->convolutional_output_layer[i - 1].con_out_layer[l].matrix[a][b];
					}
				}
				conv_mask->mask[i].weight[j][k] -= (double)(b.net_setting.learning_rate * tmp);
				tmp = 0;
				//tmp2 += train->convolutional_output_layer[i].con_out_layer[k].matrix_deviation[a][b];
			}
		}
		//train->convolutional_layer[i].filter_bias[k] -= tmp2;
	}
}

void cal_cnn_mask_bias_update(NNET &b, CNN_OUTPUT *output, CNN_MASK *conv_mask)
{
	int i, j, k;
	double tmp = 0;
		for (i = 0; i < conv_mask->mask_num; i++) {
			for (j = 0; j < output->row; j++) {
				for (k = 0; k < output->column; k++) {
					tmp += output->conv_out_layer[i].matrix_deviation[j][k];
				}
				
			}
			conv_mask->mask_bias[i] -= (double)(b.net_setting.learning_rate * tmp);
			tmp = 0;
		}
		
}