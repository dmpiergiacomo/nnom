/*
 * Testing L2 Normalize layer.
 *
 ***********************************************************************
 * EXPECTED IO VALUES:
 *
 * f32 in:
 * 7.000000, 8.000000, 13.000000, 11.000000
 *
 * f32 normalized in:
 * 0.000000, 0.166667,  1.000000, 0.666667
 *
 * Q7 normalized in:
 *        0,       21,       127,       85
 *
 * Q7 out:
 *        0,       17,       105,       71
 *
 ***********************************************************************
 *
 * INNER OPERATIONS (between last 2 layers):
 *
 * f32 normalized in:
 * 0.0000000, 0.1640625, 0.9921875, 0.6640625
 *
 * squared l2 norm:
 * (0^2)+(0.1640625^2)+(0.9921875^2)+(0.6640625^2) = 1.452331543
 *
 * l2 norm:
 * sqrt(1.452331543) = 1.20512719
 *
 * f32 out:
 * 0.000000000, 0.136137083, 0.823305215, 0.551031049
 *
 * Q7 out:
 *        0,       17,       105,       71
 *
 ***********************************************************************
 * */
#include <stdio.h>
#include <stdint.h>

#include "arm_math.h"

#include "nnom.h"
#include "weights.h"


nnom_model_t *model;

// Normalize signal within range [0, 1]
void normalize(float32_t *ppg_in, float32_t *ppg_out, uint32_t size)
{
	float32_t min;
	uint32_t min_idx;
	float32_t max;
	uint32_t max_idx;

	arm_min_f32(ppg_in, size, &min, &min_idx);
	arm_max_f32(ppg_in, size, &max, &max_idx);

	for(int i = 0; i < size; i++)
	{
		ppg_out[i] = (ppg_in[i] - min) / (max - min);
	}

	printf1(TAG_GREEN, "Signal was normalized\n");
}

int main()
{
	uint32_t nnom_in_size = 4;
	uint32_t nnom_out_size = 4;
	float32_t nnom_in[4] = {7., 8., 13., 11.};
	float32_t nnom_in_norm[4];
	// float32_t nnom_out[4] = {0.34869484, 0.39850839, 0.64757613, 0.54794903};

	printf("Testing L2 Normalize layer...\n");

	model = nnom_model_create();

	printf("nnom_in_size: %ld\n", nnom_in_size);
	printf("nnom_in:\n");
	for(int i = 0; i < nnom_in_size; i++)
	{
		printf("%f, ", nnom_in[i]);
	}
	printf("\n");

	// Normalize input
	normalize(nnom_in, nnom_in_norm, nnom_in_size);
	printf("nnom_in_norm:\n");
	for(int i = 0; i < nnom_in_size; i++)
	{
		printf("%f, ", nnom_in_norm[i]);
	}
	printf("\n");

	// Casting to Q7
	// In order to apply rounding, CMSIS DSP library should be rebuilt with
	// the ROUNDING macro defined.
	arm_float_to_q7(nnom_in, nnom_input_data, nnom_in_size);

	printf("nnom_input_data:\n");
	for(int i = 0; i < nnom_in_size; i++)
	{
		printf("%d, ", nnom_input_data[i]);
	}
	printf("\n");

	// Run one prediction
	model_run(model);

	printf("nnom_output_data:\n");
	for(int i = 0; i < nnom_out_size; i++)
	{
		printf("%d, ", nnom_output_data[i]);
	}
	printf("\n");

   return 0;
}
