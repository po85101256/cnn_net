#define _CRT_SECURE_NO_WARNINGS

#include "Header.h"
#include <stdio.h>
//#include <tchar.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


using namespace std;

void NNET::load_setting()
{
	int i, j;
	int layer_setting_num = 3;
	int convolutional_setting_num = 7;
	FILE *setting;
	setting = fopen("net_setting.txt", "r");
	fscanf(setting, "%s\n", &this->net_setting.train_data_name[0]);
	fscanf(setting, "data=%d\n", &this->net_setting.input_data_num);
	fscanf(setting, "%s\n", &this->net_setting.predict_data_name[0]);
	fscanf(setting, "data=%d\n", &this->net_setting.predict_data_num);
	fscanf(setting, "input_matrix_row=%d\n", &net_setting.input_matrix_row);
	fscanf(setting, "input_matrix_column=%d\n", &net_setting.input_matrix_column);
	fscanf(setting, "input_matrix_channel=%d\n", &net_setting.input_matrix_channel);
	fscanf(setting, "generation=%d\n", &net_setting.generation);
	fscanf(setting, "learning_rate=%lf\n", &net_setting.learning_rate);
	fscanf(setting, "learning_rate_decay_generation=%d\n", &this->net_setting.learning_rate_decay_generation);
	fscanf(setting, "learning_rate_decay=%lf\n", &this->net_setting.learning_rate_decay);
	fscanf(setting, "is_drop=%d\n", &this->net_setting.is_drop);
	fscanf(setting, "ann_layer=%d\n", &this->net_setting.layers_num);
	///
	//this->net_setting.ann_layers_setting = NULL;
	this->net_setting.ann_layers_setting = new int*[this->net_setting.layers_num];
	for (i = 0; i < this->net_setting.layers_num; i++)
	{
		this->net_setting.ann_layers_setting[i] = new int[layer_setting_num];
	}

	for (i = 0; i < this->net_setting.layers_num; i++) {

		fscanf(setting, "(");
		for (j = 0; j < layer_setting_num; j++) {
			fscanf(setting, "%d,", &this->net_setting.ann_layers_setting[i][j]);
		}
		fscanf(setting, ")\n");
	}
	///
	fscanf(setting, "\n");
	this->net_setting.input_nodes_num = this->net_setting.ann_layers_setting[0][0];
	this->net_setting.output_nodes_num = this->net_setting.ann_layers_setting[this->net_setting.layers_num - 1][0];
	fscanf(setting, "convolutional_layer=%d\n", &this->net_setting.convolutional_layer_num);
	this->net_setting.convolutional_layer_setting = new int *[this->net_setting.convolutional_layer_num];
	for (i = 0; i < this->net_setting.convolutional_layer_num; i++) {
		this->net_setting.convolutional_layer_setting[i] = new int[convolutional_setting_num];
	}
	for (i = 0; i < this->net_setting.convolutional_layer_num; i++) {

		fscanf(setting, "(");
		for (j = 0; j < convolutional_setting_num; j++) {
			fscanf(setting, "%d,", &this->net_setting.convolutional_layer_setting[i][j]);
			//printf("%d,", this->net_setting.convolutional_layer_setting[i][j]);
		}
		fscanf(setting, ")\n");
	}
	/*if (this->net_setting.convolutional_layer_setting[this->net_setting.convolutional_layer_num - 1][5] == 0) {
		this->net_setting.layers_setting[0][0] = this->net_setting.convolutional_layer_setting[this->net_setting.convolutional_layer_num - 1][0];
		this->net_setting.input_nodes_num = this->net_setting.convolutional_layer_setting[this->net_setting.convolutional_layer_num - 1][0];
	}
	else if (this->net_setting.convolutional_layer_setting[this->net_setting.convolutional_layer_num - 1][5] == 1) {
		if (this->net_setting.convolutional_layer_num - 2 >= 0) {
			this->net_setting.convolutional_layer_setting[this->net_setting.convolutional_layer_num - 1][0] = this->net_setting.convolutional_layer_setting[this->net_setting.convolutional_layer_num - 2][0];
		}
		else {
			this->net_setting.convolutional_layer_setting[this->net_setting.convolutional_layer_num - 1][0] = 1;
		}
		this->net_setting.layers_setting[0][0] = this->net_setting.convolutional_layer_setting[this->net_setting.convolutional_layer_num - 1][0];
		this->net_setting.input_nodes_num = this->net_setting.convolutional_layer_setting[this->net_setting.convolutional_layer_num - 1][0];
	}*/
	//printf("%d,", this->net_setting.input_nodes_num);
	fclose(setting);
}

void NNET::ini_output_layer()
{
	this->is_load_output = 1;
	this->softmax_layer.node_num = this->net_setting.output_nodes_num;
	this->softmax_layer.softmax_value = new double[this->net_setting.output_nodes_num];
}

void NNET::ini_weight()
{
	int i, j;
	this->is_load_weight = 1;
	this->weight = new ANN_WEIGHT[this->net_setting.layers_num - 1];
	for (i = 0; i<(net_setting.layers_num - 1); i++) {
		this->weight[i].front_layer = this->net_setting.ann_layers_setting[i][0];
		this->weight[i].behind_layer = this->net_setting.ann_layers_setting[i + 1][0];
	}

	for (i = 0; i<(this->net_setting.layers_num - 1); i++) {
		this->weight[i].weight_value = new double*[this->weight[i].front_layer];
		for (j = 0; j<this->weight[i].front_layer; j++) {
			this->weight[i].weight_value[j] = new double[this->weight[i].behind_layer];
		}
	}
}

void NNET::ini_layer()
{
	this->is_load_layer = 1;
	this->layer = new ANN_LAYER[this->net_setting.layers_num];
	for (int i = 0; i<this->net_setting.layers_num; i++) {
		layer[i].bias = new double[this->net_setting.ann_layers_setting[i][0]];
		layer[i].transfer_sum = new double[this->net_setting.ann_layers_setting[i][0]];
		layer[i].deviation_value = new double[this->net_setting.ann_layers_setting[i][0]];
		layer[i].is_drop = new int[this->net_setting.ann_layers_setting[i][0]];
		layer[i].nodes_num = this->net_setting.ann_layers_setting[i][0];
		layer[i].activation_function = this->net_setting.ann_layers_setting[i][1];
		layer[i].drop_prob = (double)(this->net_setting.ann_layers_setting[i][2]/10);
	}
}

void NNET::random_fill_model()
{
	double rand_num;
	srand((int)(time(NULL)));
	rand_num = (rand() % 21) - 10;
	int i, j, k, l;

	for (i = 0; i<this->net_setting.layers_num; i++) {
		for (j = 0; j<this->layer[i].nodes_num; j++) {
			rand_num = (double)(((rand() % 21) - 10) / 10.0);
			//rand_num = 1;
			this->layer[i].bias[j] = rand_num;
		}
	}

	for (i = 0; i<(this->net_setting.layers_num - 1); i++) {
		for (j = 0; j<this->weight[i].front_layer; j++) {
			for (k = 0; k<this->weight[i].behind_layer; k++) {
				rand_num = (double)(((rand() % 21) - 10) / 10.0);
				//rand_num = 1;
				this->weight[i].weight_value[j][k] = rand_num;
			}
		}
	}

	for (i = 0; i < this->net_setting.convolutional_layer_num; i++) {
		if (this->conv_layer[i].layer_type == 0) {
			for (j = 0; j < this->conv_layer[i].mask_num; j++) {
				for (k = 0; k < this->conv_layer[i].mask[j].row; k++) {
					for (l = 0; l < this->conv_layer[i].mask[j].column; l++) {
						rand_num = (double)(((rand() % 21) - 10) / 10.0);
						//rand_num = 1;
						this->conv_layer[i].mask[j].weight[k][l] = rand_num;
					}
				}
				rand_num = (double)(((rand() % 21) - 10) / 10.0);
				//rand_num = 1;
				this->conv_layer[i].mask_bias[j] = rand_num;
			}
		}

	}

}

void DATA_SET::load_data(NNET &b)
{
	int i, j, k;
	int input_data_long = b.net_setting.input_matrix_row* b.net_setting.input_matrix_column;
	this->data_layer = new data_value[b.net_setting.input_data_num];
	//
	for (i = 0; i<b.net_setting.input_data_num; i++) {
		this->data_layer[i].input_data = new double[input_data_long];
		this->data_layer[i].output_data = new double[b.net_setting.output_nodes_num];
	}
	//
	FILE *data;
	data = fopen(b.net_setting.train_data_name, "r");
	for (i = 0; i<b.net_setting.input_data_num; i++) {
		for (j = 0; j<input_data_long; j++) {
			fscanf(data, "%lf,", &this->data_layer[i].input_data[j]);
		}
		for (k = 0; k<b.net_setting.output_nodes_num; k++) {
			fscanf(data, "%lf,", &this->data_layer[i].output_data[k]);
		}
		fscanf(data, "\n");
	}
	fclose(data);
}

void DATA_SET::load_data(char *data_name, int data_num, NNET &b)
{
	int i, j, k;
	int input_data_long = b.net_setting.input_matrix_row* b.net_setting.input_matrix_column;
	this->data_layer = new data_value[data_num];
	//
	for (i = 0; i<b.net_setting.input_data_num; i++) {
		this->data_layer[i].input_data = new double[input_data_long];
		this->data_layer[i].output_data = new double[b.net_setting.output_nodes_num];
	}
	//
	FILE *data;
	data = fopen(data_name, "r");
	for (i = 0; i<data_num; i++) {
		for (j = 0; j<input_data_long; j++) {
			fscanf(data, "%lf,", &this->data_layer[i].input_data[j]);
		}
		for (k = 0; k<b.net_setting.output_nodes_num; k++) {
			fscanf(data, "%lf,", &this->data_layer[i].output_data[k]);
		}
		fscanf(data, "\n");
	}
	fclose(data);
}



void::NNET::ini_convolutional_layer()
{
	int i, j, k;
	this->is_load_convolutional_layer = 1;
	this->conv_layer = new CNN_MASK[this->net_setting.convolutional_layer_num];
	for (i = 0; i < this->net_setting.convolutional_layer_num; i++) {
		this->conv_layer[i].mask_num = this->net_setting.convolutional_layer_setting[i][0];
		this->conv_layer[i].mask_row = this->net_setting.convolutional_layer_setting[i][1];
		this->conv_layer[i].mask_column = this->net_setting.convolutional_layer_setting[i][2];
		this->conv_layer[i].left_step = this->net_setting.convolutional_layer_setting[i][3];
		this->conv_layer[i].down_step = this->net_setting.convolutional_layer_setting[i][4];
		this->conv_layer[i].layer_type = this->net_setting.convolutional_layer_setting[i][5];
		this->conv_layer[i].activation_function_type = this->net_setting.convolutional_layer_setting[i][6];

		if (this->conv_layer[i].layer_type == 0) {
			this->conv_layer[i].mask = new MASK[this->conv_layer[i].mask_num];
			this->conv_layer[i].mask_bias = new double[this->conv_layer[i].mask_num];
			for (j = 0; j < this->conv_layer[i].mask_num; j++) {
				this->conv_layer[i].mask[j].column = this->conv_layer[i].mask_column;
				this->conv_layer[i].mask[j].row = this->conv_layer[i].mask_row;
				//printf("|%d,%d|", this->convolutional_layer[i].filter[j].column, this->convolutional_layer[i].filter[j].row);
				this->conv_layer[i].mask[j].weight = new double *[this->conv_layer[i].mask[j].row];
				for (k = 0; k < this->conv_layer[i].mask[j].row; k++) {
					this->conv_layer[i].mask[j].weight[k] = new double[this->conv_layer[i].mask[j].column];
				}
			}
			this->conv_layer[i].inverse_mask = new double *[this->conv_layer[i].mask_row];
			for (j = 0; j < this->conv_layer[i].mask_row; j++) {
				this->conv_layer[i].inverse_mask[j] = new double[this->conv_layer[i].mask_column];
			}
		}
		else if (this->conv_layer[i].layer_type == 1) {

		}
	}

}

void::NNET::ini_convolutional_output_layer() {
	int i, j, k;
	this->conv_output = new CNN_OUTPUT[this->net_setting.convolutional_layer_num];
	for (i = 0; i < this->net_setting.convolutional_layer_num; i++) {
		if (i == 0) {
			if (this->conv_layer[i].layer_type == 0) {
				this->conv_output[i].layer_num = this->net_setting.convolutional_layer_setting[i][0];
				this->conv_output[i].column = (int)((this->net_setting.input_matrix_column - this->net_setting.convolutional_layer_setting[i][2]) / this->net_setting.convolutional_layer_setting[i][3]) + 1;
				this->conv_output[i].row = (int)((this->net_setting.input_matrix_row - this->net_setting.convolutional_layer_setting[i][1]) / this->net_setting.convolutional_layer_setting[i][4]) + 1;
				this->conv_output[i].conv_out_layer = new CNN_OUTPUT_LAYER[this->conv_output[i].layer_num];
				for (j = 0; j < this->conv_output[i].layer_num; j++) {
					this->conv_output[i].conv_out_layer[j].row = this->conv_output[i].row;
					this->conv_output[i].conv_out_layer[j].column = this->conv_output[i].column;
					this->conv_output[i].conv_out_layer[j].matrix = new double *[this->conv_output[i].conv_out_layer[j].row];
					this->conv_output[i].conv_out_layer[j].matrix_deviation = new double *[this->conv_output[i].conv_out_layer[j].row];
					for (k = 0; k < this->conv_output[i].conv_out_layer[j].row; k++) {
						this->conv_output[i].conv_out_layer[j].matrix[k] = new double[this->conv_output[i].conv_out_layer[j].column];
						this->conv_output[i].conv_out_layer[j].matrix_deviation[k] = new double[this->conv_output[i].conv_out_layer[j].column];
					}
				}
			}
			else if (this->conv_layer[i].layer_type == 1) {
				this->conv_output[i].layer_num = 1;
				this->conv_output[i].row = (int)(this->conv_output[i - 1].row / this->conv_layer[i].mask_row);
				this->conv_output[i].column = (int)(this->conv_output[i - 1].column / this->conv_layer[i].mask_column);
				this->conv_output[i].conv_out_layer = new CNN_OUTPUT_LAYER[this->conv_output[i].layer_num];
				for (j = 0; j < this->conv_output[i].layer_num; j++) {
					this->conv_output[i].conv_out_layer[j].row = this->conv_output[i].row;
					this->conv_output[i].conv_out_layer[j].column = this->conv_output[i].column;
					this->conv_output[i].conv_out_layer[j].matrix = new double *[this->conv_output[i].conv_out_layer[j].row];
					this->conv_output[i].conv_out_layer[j].matrix_deviation = new double *[this->conv_output[i].conv_out_layer[j].row];
					for (k = 0; k < this->conv_output[i].conv_out_layer[j].row; k++) {
						this->conv_output[i].conv_out_layer[j].matrix[k] = new double[this->conv_output[i].conv_out_layer[j].column];
						this->conv_output[i].conv_out_layer[j].matrix_deviation[k] = new double[this->conv_output[i].conv_out_layer[j].column];
					}
				}
			}

		}
		else {
			if (this->conv_layer[i].layer_type == 0) {
				this->conv_output[i].layer_num = this->net_setting.convolutional_layer_setting[i][0];
				//printf("$%d", this->convolutional_output_layer[i].layer_num);
				this->conv_output[i].column = (int)((this->conv_output[i - 1].row - this->net_setting.convolutional_layer_setting[i][2]) / this->net_setting.convolutional_layer_setting[i][3]) + 1;
				this->conv_output[i].row = (int)((this->conv_output[i - 1].column - this->net_setting.convolutional_layer_setting[i][1]) / this->net_setting.convolutional_layer_setting[i][4]) + 1;
				this->conv_output[i].conv_out_layer = new CNN_OUTPUT_LAYER[this->conv_output[i].layer_num];
				for (j = 0; j < this->conv_output[i].layer_num; j++) {
					this->conv_output[i].conv_out_layer[j].row = this->conv_output[i].row;
					this->conv_output[i].conv_out_layer[j].column = this->conv_output[i].column;
					this->conv_output[i].conv_out_layer[j].matrix = new double *[this->conv_output[i].conv_out_layer[j].row];
					this->conv_output[i].conv_out_layer[j].matrix_deviation = new double *[this->conv_output[i].conv_out_layer[j].row];
					for (k = 0; k < this->conv_output[i].conv_out_layer[j].row; k++) {
						this->conv_output[i].conv_out_layer[j].matrix[k] = new double[this->conv_output[i].conv_out_layer[j].column];
						this->conv_output[i].conv_out_layer[j].matrix_deviation[k] = new double[this->conv_output[i].conv_out_layer[j].column];
					}

				}
			}
			else if (this->conv_layer[i].layer_type == 1) {
				this->conv_output[i].layer_num = this->net_setting.convolutional_layer_setting[i - 1][0];
				this->conv_output[i].row = (int)(this->conv_output[i - 1].row / this->conv_layer[i].mask_row);
				this->conv_output[i].column = (int)(this->conv_output[i - 1].column / this->conv_layer[i].mask_column);
				this->conv_output[i].conv_out_layer = new CNN_OUTPUT_LAYER[this->conv_output[i].layer_num];
				for (j = 0; j < this->conv_output[i].layer_num; j++) {
					this->conv_output[i].conv_out_layer[j].row = this->conv_output[i].row;
					this->conv_output[i].conv_out_layer[j].column = this->conv_output[i].column;
					this->conv_output[i].conv_out_layer[j].matrix = new double *[this->conv_output[i].conv_out_layer[j].row];
					this->conv_output[i].conv_out_layer[j].matrix_deviation = new double *[this->conv_output[i].conv_out_layer[j].row];
					for (k = 0; k < this->conv_output[i].conv_out_layer[j].row; k++) {
						this->conv_output[i].conv_out_layer[j].matrix[k] = new double[this->conv_output[i].conv_out_layer[j].column];
						this->conv_output[i].conv_out_layer[j].matrix_deviation[k] = new double[this->conv_output[i].conv_out_layer[j].column];
					}
				}
			}

		}
	}
}

void NNET::save_model(char *modelt)
{
	int i, j, k,l;
	int layer_setting_num = 3;
	int convolutional_setting_num = 7;
	FILE *model;
	model = fopen(modelt, "w");
	fprintf(model, "input_matrix_row=%d\n", this->net_setting.input_matrix_row);
	fprintf(model, "input_matrix_column=%d\n", this->net_setting.input_matrix_column);
	fprintf(model, "input_matrix_channel=%d\n", this->net_setting.input_matrix_channel);
	fprintf(model, "output=%d\n", this->net_setting.output_nodes_num);
	fprintf(model, "ann_layer=%d\n", this->net_setting.layers_num);
	for (i = 0; i < this->net_setting.layers_num; i++) {
		fprintf(model, "(");
		for (j = 0; j<layer_setting_num; j++) {
			fprintf(model, "%d,", this->net_setting.ann_layers_setting[i][j]);
		}
		fprintf(model, ")\n");
	}
	fprintf(model, "\n");
	///
	fprintf(model, "cnn_layer=%d\n", this->net_setting.convolutional_layer_num);
	for (i = 0; i < this->net_setting.convolutional_layer_num; i++) {
		fprintf(model, "(");
		for (j = 0; j<convolutional_setting_num; j++) {
			fprintf(model, "%d,", this->net_setting.convolutional_layer_setting[i][j]);
		}
		fprintf(model, ")\n");
	}
	fprintf(model, "\n");
	///conv
	fprintf(model, "\ncnn_mask:\n");
	for (i = 0; i < this->net_setting.convolutional_layer_num; i++) {
		if (this->conv_layer[i].layer_type == 0) {
			fprintf(model, "conv_layer:%d,layer_type:%d\n",i, this->conv_layer[i].layer_type);
			for (j = 0; j < this->conv_layer[i].mask_num; j++) {
				for (k = 0; k < this->conv_layer[i].mask[j].row; k++) {
					for (l = 0; l < this->conv_layer[i].mask[j].column; l++) {
						fprintf(model, "%.10f,", this->conv_layer[i].mask[j].weight[k][l]);
					}
					fprintf(model, "\n");
				}
				fprintf(model, "cnn_bias:%.10f\n", this->conv_layer[i].mask_bias[j]);
			}
		}
		else if (this->conv_layer[i].layer_type == 1) {
			fprintf(model, "conv_layer:%d,layer_type:%d\n", i, this->conv_layer[i].layer_type);
		}
	}
	///
	fprintf(model, "\nann_node_bias:\n");
	for (i = 1; i<(this->net_setting.layers_num); i++) {
		for (j = 0; j<this->net_setting.ann_layers_setting[i][0]; j++) {
			fprintf(model, "%.10f,", this->layer[i].bias[j]);
		}
		fprintf(model, "\n");
	}
	fprintf(model, "\nann_weight:\n");
	for (i = 0; i<this->net_setting.layers_num - 1; i++) {
		for (j = 0; j<this->weight[i].front_layer; j++) {
			for (k = 0; k<this->weight[i].behind_layer; k++) {
				fprintf(model, "%.10f,", this->weight[i].weight_value[j][k]);
			}
			fprintf(model, "\n");
		}
		fprintf(model, "\n");
	}
	fclose(model);
}
void::NNET::load_model(char *modelt)
{

	int i, j, k, l;
	int layer_setting_num = 3;
	int convolutional_setting_num = 7;
	FILE *model;
	int tmp;
	double tmp2;
	model = fopen(modelt, "r");
	fscanf(model, "input_matrix_row=%d\n", &this->net_setting.input_matrix_row);
	fscanf(model, "input_matrix_column=%d\n", &this->net_setting.input_matrix_column);
	fscanf(model, "input_matrix_channel=%d\n", &this->net_setting.input_matrix_channel);
	fscanf(model, "output=%d\n", &this->net_setting.output_nodes_num);
	fscanf(model, "ann_layer=%d\n", &this->net_setting.layers_num);
	////
	if (this->net_setting.ann_layers_setting == NULL) {
		this->net_setting.ann_layers_setting = new int*[this->net_setting.layers_num];
		for (i = 0; i < this->net_setting.layers_num; i++)
		{
			this->net_setting.ann_layers_setting[i] = new int[layer_setting_num];
		}
	}
	
	////
	for (i = 0; i < this->net_setting.layers_num; i++) {
		fscanf(model, "(");
		for (j = 0; j<layer_setting_num; j++) {
			fscanf(model, "%d,", &this->net_setting.ann_layers_setting[i][j]);
		}
		fscanf(model, ")\n");
	}
	fscanf(model, "\n");

	////
	fscanf(model, "cnn_layer=%d\n", &this->net_setting.convolutional_layer_num);
	this->net_setting.convolutional_layer_setting = new int *[this->net_setting.convolutional_layer_num];
	for (i = 0; i < this->net_setting.convolutional_layer_num; i++) {
		this->net_setting.convolutional_layer_setting[i] = new int[convolutional_setting_num];
	}
	////
	
	for (i = 0; i < this->net_setting.convolutional_layer_num; i++) {
		fscanf(model, "(");
		for (j = 0; j<convolutional_setting_num; j++) {
			fscanf(model, "%d,", &this->net_setting.convolutional_layer_setting[i][j]);
		}
		fscanf(model, ")\n");
	}
	fscanf(model, "\n");
	////
	///--
	if (this->layer == NULL) {
		this->ini_layer();
	}
	if (this->weight == NULL) {
		this->ini_weight();
	}
	if (this->is_load_output != 1) {
		this->ini_output_layer();
	}
	if (this->conv_layer == NULL) {
		this->ini_convolutional_layer();
	}
	if (this->conv_output == NULL) {
		this->ini_convolutional_output_layer();
	}
	///--
	fscanf(model, "\ncnn_mask:\n");
	for (i = 0; i < this->net_setting.convolutional_layer_num; i++) {
		if (this->conv_layer[i].layer_type == 0) {
			fscanf(model, "conv_layer:%d,layer_type:%d\n", &tmp, &this->conv_layer[i].layer_type);
			for (j = 0; j < this->conv_layer[i].mask_num; j++) {
				for (k = 0; k < this->conv_layer[i].mask[j].row; k++) {
					for (l = 0; l < this->conv_layer[i].mask[j].column; l++) {
						fscanf(model, "%lf,", &this->conv_layer[i].mask[j].weight[k][l]);
					}
					fscanf(model, "\n");
				}
				fscanf(model, "cnn_bias:%lf\n", &this->conv_layer[i].mask_bias[j]);
			}
		}
		else if (this->conv_layer[i].layer_type == 1) {
			fscanf(model, "conv_layer:%d,layer_type:%d\n", &tmp, &this->conv_layer[i].layer_type);
		}
	}
	///
	fscanf(model, "\nann_node_bias:\n");
	for (i = 1; i<(this->net_setting.layers_num); i++) {
		for (j = 0; j<this->net_setting.ann_layers_setting[i][0]; j++) {
			fscanf(model, "%lf,", &this->layer[i].bias[j]);
		}
		fscanf(model, "\n");
	}
	fscanf(model, "\nann_weight:\n");
	for (i = 0; i<this->net_setting.layers_num - 1; i++) {
		for (j = 0; j<this->weight[i].front_layer; j++) {
			for (k = 0; k<this->weight[i].behind_layer; k++) {
				fscanf(model, "%lf,", &this->weight[i].weight_value[j][k]);
			}
			fscanf(model, "\n");
		}
		fscanf(model, "\n");
	}
	fclose(model);

}