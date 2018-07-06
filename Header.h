//#pragma once
#ifndef HEADER_H
#define HEADER_H

#include <time.h>
#include <math.h>

class ANN_LAYER
{
public:
	double *deviation_value = NULL;
	double *bias = NULL;
	double *transfer_sum = NULL;
	int *is_drop = NULL;
	int activation_function;
	double drop_prob;
	int nodes_num;
	~ANN_LAYER() {
		delete[] bias;
		delete[] deviation_value;
		delete[] transfer_sum;
		delete[] is_drop;
		bias = NULL;
		deviation_value = NULL;
		transfer_sum = NULL;
		is_drop = NULL;
	}
};

class ANN_WEIGHT
{
public:
	double **weight_value = NULL;
	int front_layer;
	int behind_layer;
	~ANN_WEIGHT() {
		int j;
		for (j = 0; j<front_layer; j++) {
			delete[] weight_value[j];
			weight_value[j] = NULL;
		}delete[]weight_value;
		weight_value = NULL;
	}
};

class ANN_SOFTMAX
{
public:
	int node_num;
	double *softmax_value = NULL;
	~ANN_SOFTMAX() {
		delete[] softmax_value;
		softmax_value = NULL;
	}
};

class NNET_setting
{
public:

	char train_data_name[50];
	char predict_data_name[50];
	int input_data_num;
	int predict_data_num;
	int input_matrix_row;
	int input_matrix_column;
	int input_nodes_num;
	int generation;
	int output_nodes_num;
	int layers_num;
	int is_drop;
	int input_matrix_channel;
	double learning_rate_decay;
	int learning_rate_decay_generation;
	int **ann_layers_setting = NULL;
	int convolutional_layer_num;
	int **convolutional_layer_setting = NULL;
	double learning_rate;

	NNET_setting() {
		convolutional_layer_num = 0;
	}
	~NNET_setting() {
		int i;
		
		if (convolutional_layer_num != 0) {
			for (i = 0; i < convolutional_layer_num; i++) {
				delete[] convolutional_layer_setting[i];
				convolutional_layer_setting[i] = NULL;
			}
			delete[] convolutional_layer_setting;
			convolutional_layer_setting = NULL;
		}
		if (ann_layers_setting != 0) {
			for (i = 0; i < layers_num; i++) {
				delete[] ann_layers_setting[i];
				ann_layers_setting[i] = NULL;
			}
			delete[] ann_layers_setting;
			ann_layers_setting = NULL;
		}
	}
};

class MASK
{
public:
	int row;
	int column;
	double **weight;
	~MASK() {
		int i;
		//printf("%d,%d", this->row, this->column);
		for (i = 0; i < this->row; i++) {
			delete[]this->weight[i];
			this->weight[i] = NULL;
		}delete[] this->weight;
		this->weight = NULL;
	}
};

class CNN_MASK
{
public:
	int layer_type;
	int mask_num;
	int left_step;
	int down_step;
	int mask_row;
	int mask_column;
	int activation_function_type;
	MASK *mask = NULL;
	double *mask_bias = NULL;
	double **inverse_mask = NULL;
	~CNN_MASK() {
		if (layer_type == 0) {
			if (mask != NULL) {
				delete[] mask;
				mask = NULL;
			}
			if (mask_bias != NULL) {
				delete[] mask_bias;
				mask_bias = NULL;
			}
			if (inverse_mask != NULL) {
				for (int i = 0; i < mask_row; i++) {
					delete[]inverse_mask[i];
				}

				delete[] inverse_mask;
				inverse_mask = NULL;
			}
		}

	}
};

class CNN_OUTPUT_LAYER
{
public:
	int row;
	int column;
	double **matrix;
	double **matrix_deviation;
	~CNN_OUTPUT_LAYER() {
		int i;
		for (i = 0; i < row; i++) {
			delete[]matrix[i];
			delete[]matrix_deviation[i];
			matrix[i] = NULL;
			matrix_deviation[i] = NULL;
		}
		delete[] matrix;
		delete[] matrix_deviation;
		matrix = NULL;
		matrix_deviation = NULL;
	}
};

class CNN_OUTPUT
{
public:
	int row;
	int column;
	int layer_num;
	CNN_OUTPUT_LAYER * conv_out_layer = NULL;
	~CNN_OUTPUT() {
		if (conv_out_layer != NULL) {
			delete[] conv_out_layer;
			conv_out_layer = NULL;
		}

	}
};

class NNET
{
public:
	NNET() {
		is_load_weight = 0;
		is_load_layer = 0;
		is_load_output = 0;
		is_load_convolutional_layer = 0;
	}
	NNET_setting net_setting;
	ANN_WEIGHT *weight = NULL;
	ANN_LAYER *layer = NULL;
	ANN_SOFTMAX softmax_layer;
	CNN_MASK *conv_layer = NULL;
	CNN_OUTPUT *conv_output = NULL;

	int is_load_weight;
	int is_load_layer;
	int is_load_output;
	int is_load_convolutional_layer;

	void load_setting();
	void ini_weight();
	void ini_layer();
	void ini_output_layer();
	void ini_convolutional_layer();
	void ini_convolutional_output_layer();
	void random_fill_model();
	void load_model(char *modelt);
	void save_model(char *modelt);
	~NNET() {
		///
		if (is_load_weight == 1) {
			delete[] this->weight;
			this->weight = NULL;
		}
		if (is_load_layer == 1) {
			delete[] this->layer;
			this->layer = NULL;
		}
		if (is_load_convolutional_layer == 1) {
			delete[] this->conv_layer;
			this->conv_layer = NULL;
			delete[] this->conv_output;
			this->conv_output = NULL;
		}
	};
};

class data_value
{
public:
	double *input_data = NULL;
	double *output_data = NULL;
	~data_value() {
		delete[] this->input_data;
		delete[] this->output_data;
		this->input_data = NULL;
		this->output_data = NULL;
	}
};

class DATA_SET
{
public:
	data_value * data_layer = NULL;
	void load_data(NNET &b);
	void load_data(char *data_name,int data_num, NNET &b);
	~DATA_SET() {
		delete[] this->data_layer;
		this->data_layer = NULL;
	}
};


#endif