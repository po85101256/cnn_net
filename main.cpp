#define _CRT_SECURE_NO_WARNINGS

#include<stdio.h>
#include "Header.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
void train_net(NNET &b, DATA_SET &d, int first, int last);
//void cal_backward(NNET *train, DATA_SET &output, int n);
void cal_forward(ANN_LAYER *input_layer, ANN_LAYER *output_layer,ANN_WEIGHT *weight);
void cal_ann_layer_forward(ANN_LAYER *input_layer, ANN_LAYER *output_layer, ANN_WEIGHT *weight, int use_drop);
void cal_cnn_layer_forward(CNN_OUTPUT *front_output, CNN_MASK *conv_mask, CNN_OUTPUT *output);
void cal_cnn_layer_sensitive_map(CNN_OUTPUT *front_output, CNN_MASK *conv_mask, CNN_OUTPUT *output);
void cal_cnn_mask_update(NNET &b,CNN_OUTPUT *front_output, CNN_MASK *conv_mask, CNN_OUTPUT *output);
void cal_cnn_mask_update_first(NNET &b,DATA_SET &d, int n, CNN_MASK *conv_mask, CNN_OUTPUT *output);
void cal_cnn_mask_bias_update(NNET &b, CNN_OUTPUT *output, CNN_MASK *conv_mask);
void cal_ann_forward_output(NNET *train, ANN_LAYER *input_layer);
void cal_deviation_value(NNET &b, ANN_SOFTMAX &softmax_layer,ANN_LAYER *output_layer ,DATA_SET &d ,int n);
void cal_backward(NNET &b, ANN_LAYER *layer, ANN_LAYER *front_layer, ANN_WEIGHT *weight, int use_drop);
void cal_weight_backward(NNET &b, ANN_LAYER *layer, ANN_LAYER *front_layer, ANN_WEIGHT *weight, int use_drop);
void set_data_to_ann(NNET &b, DATA_SET &d, int n);
void set_data_to_cnn(NNET &b, DATA_SET &d, int n);
void ann_cal_forward(NNET &b, DATA_SET &d, int n);
void ann_cal_backward(NNET &b, DATA_SET &d, int n);
void cnn_cal_forward(NNET &b, DATA_SET &d, int n);
void cnn_cal_backward(NNET &b, DATA_SET &d, int n);

void printf_cnn(NNET &b);
void printf_ann(NNET &b);
void printf_ann_b(NNET &b);

void cnn_to_ann(NNET &b);
void ann_to_cnn(NNET &b);
void determine_is_drop(NNET &b);
double cost_function(double actual_value, double output_value);
void cal_data_cnn(NNET &b, DATA_SET &d, int n);
void cal_predict_rate(NNET &b, DATA_SET &d, int inputdata_num);
void train();
void pridict();

int main()
{
	train();
	pridict();
	return 0;
}
void pridict()
{
	int i, j, k;
	double tmp2 = 0;
	char tt[25] = "cnn_model.txt";
	NNET b;
	b.load_setting();
	b.load_model(tt);
	b.net_setting.is_drop = 0;
	//b.ini_layer();
	//b.ini_output_layer();
	//b.ini_weight();

	//b.ini_convolutional_layer();
	//b.ini_convolutional_output_layer();
	//b.random_fill_model();

	//printf_cnn(b);
	char dd[25] = "iris2.txt";
	int input_data_num = 150;
	DATA_SET train_data;
	train_data.load_data(dd,input_data_num,b);
	//train_data.load_data(b);

			for (j = 0; j<input_data_num; j++) {
				cnn_cal_forward(b, train_data, j);
				cnn_to_ann(b);
				ann_cal_forward(b, train_data, j);
				for (k = 0; k < b.net_setting.output_nodes_num; k++) {
					tmp2 += (train_data.data_layer[j].output_data[k] - b.softmax_layer.softmax_value[k])* (
						train_data.data_layer[j].output_data[k] - b.softmax_layer.softmax_value[k]);
				}

			}
			//printf("%d,%lf\n", i, (double)0.5*tmp2);
			tmp2 = 0;

	cal_predict_rate(b, train_data, b.net_setting.input_data_num);
	///
	FILE *result;
	result = fopen("result.txt", "w");
	for (i = 0; i<b.net_setting.input_data_num; i++) {
		cnn_cal_forward(b, train_data, i);
		cnn_to_ann(b);
		ann_cal_forward(b, train_data, i);
		for (j = 0; j<b.net_setting.output_nodes_num; j++) {
			fprintf(result, "%f,", b.softmax_layer.softmax_value[j]);
		}
		fprintf(result, "\n");
	}

	fclose(result);
	///
	//char tt[25] = "cnn_model.txt";
	//b.save_model(tt);
	///
	system("pause");
}
void train() 
{
	int i, j, k;
	double tmp2 = 0;
	NNET b;
	b.load_setting();

	b.ini_layer();
	b.ini_output_layer();
	b.ini_weight();

	b.ini_convolutional_layer();
	b.ini_convolutional_output_layer();
	b.random_fill_model();

	//printf_cnn(b);
	DATA_SET train_data;
	train_data.load_data(b);

	//for (i = 0; i < 1; i++) {
	for (i = 0; i < b.net_setting.generation; i++) {

		train_net(b, train_data, 0, b.net_setting.input_data_num);

		if ((i % 100) == 0) {
			for (j = 0; j<b.net_setting.input_data_num; j++) {
				cnn_cal_forward(b, train_data, j);
				cnn_to_ann(b);
				ann_cal_forward(b, train_data, j);
				for (k = 0; k < b.net_setting.output_nodes_num; k++) {
					tmp2 += (train_data.data_layer[j].output_data[k] - b.softmax_layer.softmax_value[k])* (
						train_data.data_layer[j].output_data[k] - b.softmax_layer.softmax_value[k]);
				}

			}
			printf("%d,%lf\n", i, (double)0.5*tmp2);
			tmp2 = 0;
		}
		if ((i % 5000) == 0) {
			b.net_setting.learning_rate = b.net_setting.learning_rate*b.net_setting.learning_rate_decay;
		}
	}
	cal_predict_rate(b, train_data, b.net_setting.input_data_num);
	///
	FILE *result;
	result = fopen("result.txt", "w");
	for (i = 0; i<b.net_setting.input_data_num; i++) {
		cnn_cal_forward(b, train_data, i);
		cnn_to_ann(b);
		ann_cal_forward(b, train_data, i);
		for (j = 0; j<b.net_setting.output_nodes_num; j++) {
			fprintf(result, "%f,", b.softmax_layer.softmax_value[j]);
		}
		fprintf(result, "\n");
	}

	fclose(result);
	///
	char tt[25] = "cnn_model.txt";
	b.save_model(tt);
	///
	system("pause");
	
}
void cal_predict_rate(NNET &b, DATA_SET &d,int inputdata_num)
{
	int i, j, k;
	double a, t;
	int true_label, predict_label;
	int counter = 0;
	double pridict_rate=0.0;
	for (j = 0; j < inputdata_num; j++) {
		cnn_cal_forward(b, d, j);
		cnn_to_ann(b);
		ann_cal_forward(b, d, j);

		a = b.softmax_layer.softmax_value[0];
		true_label = 0;
		t = d.data_layer[j].output_data[0];
		predict_label = 0;
		for (k = 0; k < b.net_setting.output_nodes_num; k++) {
			if (b.softmax_layer.softmax_value[k] >= a) {
				a = b.softmax_layer.softmax_value[k];
				true_label = k;
			}
			if (d.data_layer[j].output_data[k] >= t) {
				t = d.data_layer[j].output_data[k];
				predict_label = k;
			}
			
		}
		if (true_label == predict_label) {
			counter++;
		}

	}
	pridict_rate = counter;
	pridict_rate= (pridict_rate/ inputdata_num);
	printf("pridict_rate:%lf(%d/%d)\n", pridict_rate, counter, inputdata_num);
}







void printf_cnn(NNET &b)
{
	int i, j, k,l;
	printf("weight:\n");
	for (i = 0; i < b.net_setting.convolutional_layer_num;i++) {
		if (b.conv_layer[i].layer_type == 0) {
			printf("conv:\n");
			for (l = 0; l < b.conv_layer[i].mask_num; l++) {
				for (j = 0; j < b.conv_layer[i].mask_row; j++) {
					for (k = 0; k < b.conv_layer[i].mask_column; k++) {
						printf("%lf,", b.conv_layer[i].mask[l].weight[j][k]);
					}
					printf("\n");

				}
				printf("bias:%lf\n", b.conv_layer[i].mask_bias[l]);
				printf("\n");

			}
			printf("\n");
		}
		else if (b.conv_layer[i].layer_type == 1) {
			printf("pool:\n\n");

		}
		
	}
	printf("output:\n");
	for (i = 0; i < b.net_setting.convolutional_layer_num; i++) {
		for (l = 0; l < b.conv_output[i].layer_num; l++) {
			for (j = 0; j < b.conv_output[i].row; j++) {
				for (k = 0; k < b.conv_output[i].row; k++) {
					printf("%lf,", b.conv_output[i].conv_out_layer[l].matrix[j][k]);
				}
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}



}

void printf_cnn_b(NNET &b)
{
	int i, j, k, l;
	printf("deviation:\n");
	/*for (i = 0; i < b.net_setting.convolutional_layer_num; i++) {
		for (l = 0; l < b.conv_layer[i].mask_num; l++) {
			for (j = 0; j < b.conv_layer[i].mask_row; j++) {
				for (k = 0; k < b.conv_layer[i].mask_column; k++) {
					printf("%lf,", b.conv_layer[i].mask[l].weight[j][k]);
				}
				printf("\n");

			}
			printf("bias:%lf\n", b.conv_layer[i].mask_bias[l]);
			printf("\n");

		}
		printf("\n");
	}*/

	for (i = 0; i < b.net_setting.convolutional_layer_num; i++) {
		for (l = 0; l < b.conv_output[i].layer_num; l++) {
			for (j = 0; j < b.conv_output[i].row; j++) {
				for (k = 0; k < b.conv_output[i].row; k++) {
					printf("%lf,", b.conv_output[i].conv_out_layer[l].matrix_deviation[j][k]);
				}
				printf("\n");
			}
			printf("\n");
		}
		printf("\n");
	}



}

void printf_ann(NNET &b)
{
	int i, j, k;
	printf("weight:\n");

	for (j = 0; j < b.net_setting.layers_num-1; j++) {
		for (i = 0; i < b.weight[j].front_layer; i++) {
			for (k = 0; k < b.weight[j].behind_layer; k++) {
				printf("%lf,", b.weight[j].weight_value[i][k]);
			}
			printf("\n");

		}
		printf("\n");
	}
	printf("bias:\n");
	for (j = 1; j < b.net_setting.layers_num; j++) {
		for (i = 0; i < b.layer[j].nodes_num; i++) {
			printf("%lf,", b.layer[j].bias[i]);
		}
		printf("\n");
	}

	printf("output:\n");
	for (j = 0; j < b.net_setting.layers_num; j++) {
		for (i = 0; i < b.layer[j].nodes_num; i++) {
			printf("%lf,", b.layer[j].transfer_sum[i]);

		}
		printf("\n");
	}

	printf("softmax:\n");
	
		for (i = 0; i < b.softmax_layer.node_num; i++) {
			printf("%lf,", b.softmax_layer.softmax_value[i]);

		}
		printf("\n");
	
}

void printf_ann_b(NNET &b)
{
	int i, j, k;
	printf("deviation:\n");
	for (j = 0; j < b.net_setting.layers_num; j++) {
		for (i = 0; i < b.layer[j].nodes_num; i++) {
			printf("%lf,", b.layer[j].deviation_value[i]);

		}
		printf("\n");
	}

	printf("updated_weight:\n");

	for (j = 0; j < b.net_setting.layers_num - 1; j++) {
		for (i = 0; i < b.weight[j].front_layer; i++) {
			for (k = 0; k < b.weight[j].behind_layer; k++) {
				printf("%lf,", b.weight[j].weight_value[i][k]);
			}
			printf("\n");

		}
		printf("\n");
	}
}