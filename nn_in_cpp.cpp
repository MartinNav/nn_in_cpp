// nn_in_cpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include<stdlib.h>
#include<math.h>

#define INPUT_COUNT 2
#define HIDDEN_NODES_COUNT 2
#define OUTPUTS_COUNT 1

#define C_TRAINING_SETS 4

const double learning_rate = 0.01;
double init_w() {
	return ((double)rand() / (double)RAND_MAX);
}

/*
void shuffle(int* arr, size_t n) {
	if (n > 1) {
		for (size_t i = 0; i < n-1; i++)
		{

		}
	}
}*/

//Using macros to avoid unnecesary fn calls
#define sigmoid(x) return 1/(1+exp(-x))
#define leaky_RELU(x) if(x>0){return x;}return x/10

#define der_sigmoid(x) return x * (1 - x)
#define der_RELU(x) if(x>0){return 1;}return 0

double activationFn(double x) {
	sigmoid(x);
}
double der_activation(double x) {
	der_sigmoid(x);
}

//https://www.youtube.com/watch?v=LA4I3cWkp1E
int main()
{

	double hiddenL[HIDDEN_NODES_COUNT];
	double outputL[OUTPUTS_COUNT];

	double hiddenBias[HIDDEN_NODES_COUNT];
	double outputBias[OUTPUTS_COUNT];

	double hiddenW[INPUT_COUNT][HIDDEN_NODES_COUNT];
	double outputW[HIDDEN_NODES_COUNT][OUTPUTS_COUNT];


	//This is because I am aproximating OR, for other functions it will be different
	double trainingI[C_TRAINING_SETS][INPUT_COUNT] = {
		{0.0,0.0},
		{0.0,1.0},
		{1.0,0.0},
		{1.0,1.0}
	};
	double trainingO[C_TRAINING_SETS][OUTPUTS_COUNT] = {
		{0.0},
		{1.0},
		{1.0},
		{1.0}
	};
	for (int i = 0; i < INPUT_COUNT; i++)
	{
		for (int j = 0; j < HIDDEN_NODES_COUNT; j++)
		{
			hiddenW[i][j] = init_w();
		}
	}
	for (int i = 0; i < HIDDEN_NODES_COUNT; i++)
	{
		for (int j = 0; j < OUTPUTS_COUNT; j++)
		{
			outputW[i][j] = init_w();
		}
		hiddenBias[i] = init_w();
	}
	for (int i = 0; i < OUTPUTS_COUNT; i++)
	{
		outputBias[i] = init_w();
	}
	int num_of_epochs = 10000;

	//Training
	for (int e = 0; e < num_of_epochs; e++)
	{
		for (int x = 0; x < C_TRAINING_SETS; x++)
		{

			//Forward pass
			//hidden
			for (int j = 0; j < HIDDEN_NODES_COUNT; j++)
			{
				double activation = hiddenBias[j];
				for (int k = 0; k < INPUT_COUNT; k++)
				{
					activation += trainingI[x][j] * hiddenW[k][j];

				}
				hiddenL[j] = activationFn(activation);
			}
			//output
			for (int j = 0; j < OUTPUTS_COUNT; j++)
			{
				double activation = outputBias[j];
				for (int k = 0; k < HIDDEN_NODES_COUNT; k++)
				{
					activation += hiddenL[k] * outputW[k][j];

				}
				outputL[j] = activationFn(activation);
			}
			//Output while training
			std::cout << "Input: "<< trainingI[x][0]<<", "<<trainingI[x][1] << " Output: " << outputL[0] << " Expected output: " << trainingO[x][0] << std::endl;
			std::cout << "err:" << (trainingO[x][0] - outputL[0]) << std::endl;
			double deltaO[OUTPUTS_COUNT];
			for (int j = 0; j < OUTPUTS_COUNT; j++)
			{
				double err = trainingO[x][j] - outputL[j];
				deltaO[j] = err * der_activation(outputL[j]);
			}
			double deltaH[HIDDEN_NODES_COUNT];
			for (int j = 0; j < HIDDEN_NODES_COUNT; j++)
			{
				double err = 0.0;
				for (int k = 0; k < OUTPUTS_COUNT; k++)
				{
					err += deltaO[k] * outputW[j][k];
				}
				deltaH[j] = err * der_activation(hiddenL[j]);
			}
			//Update output W
			for (int j = 0; j < OUTPUTS_COUNT; j++)
			{
				outputBias[j] += deltaO[j] * learning_rate;
				for (int k = 0; k < HIDDEN_NODES_COUNT; k++)
				{
					outputW[k][j] += hiddenL[k] * deltaO[j] * learning_rate;
				}
			}
			//Update hidden W
			for (int j = 0; j < HIDDEN_NODES_COUNT; j++)
			{
				hiddenBias[j] += deltaH[j] * learning_rate;
				for (int k = 0; k < INPUT_COUNT; k++)
				{
					hiddenW[k][j] += hiddenL[k] * deltaH[j] * learning_rate;
				}
			}
		
		}

	}
}


