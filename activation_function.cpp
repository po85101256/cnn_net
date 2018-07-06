#define _CRT_SECURE_NO_WARNINGS

#include "Header.h"
#include <stdio.h>
//#include <tchar.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double activation_function(double value,int function)
{
	/*  double aa=(1.0+exp(-value));
	double bb=(double)1.0/aa;
	return bb;*/
	double aa = 0;
	switch (function)
	{
	case 1:
	{
		if (value >= 0) {
			aa = value;
		}
		else if (value<0) {
			aa = 0.01*value;
			 //aa=0.0;
		}
	}
		break;
	default:
		break;
	}
	
	return aa;
}
double activation_d_function(double value, int function)
{
	// return (double)(value*(1.0-value));
	double aa = 0;

	switch (function)
	{
	case 1:
	{
		if (value >= 0) {
			aa = 1.0;
		}
		else if (value<0) {
			aa = 0.01;
			//aa=0.0;
		}
	}
	break;
	default:
		break;
	}


	
	return aa;
}

double cost_d_function(double actual_value, double output_value)
{
	//return (double)((actual_value)*log(0.5*(output_value+1)));
	return (double)(output_value - actual_value);
}

double cost_function(double actual_value, double output_value)
{
	double tmp = 0;
	tmp = (double)0.5*(actual_value - output_value)*(actual_value - output_value);
	return tmp;
}