/*
 * Implemenntation of the L2 normalization layer.
 *
 * In the current implementation a workaround is in place to retrieve the
 * float32_t layer output value. To enable such, a float32_t array needs to be
 * passed as parameters value to the nnom_lambda_config_t structure.
 */
#include "layers/nnom_l2_normalize.h"
#include "nnom.h"
#include "arm_math.h"


// TODO: This line must be removed if compiling with CMSIS source
#define F32_MIN   (-FLT_MAX)


nnom_status_t l2norm_build(nnom_layer_t *layer)
{
	// get the last layer's output as input shape
	layer->in->tensor = layer->in->hook.io->tensor;

	// output tensor
	// 1. allocate a new tensor for output
	// 2. set the same dim, qfmt to the new tensor.
	layer->out->tensor = new_tensor(NNOM_QTYPE_PER_TENSOR,layer->in->tensor->num_dim, tensor_get_num_channel(layer->in->tensor));
	tensor_cpy_attr(layer->out->tensor, layer->in->tensor);

	// see if the activation will change the q format
	if(layer->actail)
		layer->out->tensor->q_dec[0] = act_get_dec_bit(layer->actail->type, layer->out->tensor->q_dec[0]);

	// now this build has passed the input tensors (shapes, formats) to the new tensors.
	return NN_SUCCESS;
}

nnom_status_t l2norm_free(nnom_layer_t *layer)
{
	return NN_SUCCESS;
}

nnom_status_t l2norm_run(nnom_layer_t *layer)
{
	size_t in_size = tensor_size_byte(layer->in->tensor);
	float32_t in_f32[in_size];
	float32_t out_f32[in_size];
	float32_t l2_inv[in_size];
	float32_t sq_sum, sq_l2, l2;
	uint32_t sq_l2_idx;

	// CMSIS DSP supports only a q7_t power operation which returns a q31_t,
	// therefore this layer is operated as f32 and the casted back to q7_t
	arm_q7_to_float(layer->in->tensor->p_data, in_f32, in_size);

	// Calculate L2-norm
	arm_power_f32(in_f32, in_size, &sq_sum);
	float32_t dnmtr[2] = {sq_sum, F32_MIN};
	arm_max_f32(dnmtr, 2, &sq_l2, &sq_l2_idx);
	arm_sqrt_f32(sq_l2, &l2);

	// L2-normalize
	for(int i = 0; i < in_size; i++)
	{
		l2_inv[i] = 1 / l2;
	}
	arm_mult_f32(in_f32, l2_inv, out_f32, in_size);

	// Convert back to q7_t
	arm_float_to_q7(out_f32, layer->out->tensor->p_data, in_size);

	// TODO: Ugly no-brain workaround to get stuff working quickly,
	//		 but needs to disappear!!!
	// Copying f32 output value directly to parameters buffer
	// for when f32 layer output is needed rather than q7_t
	float32_t * nnom_output_data_f32 = layer->parameters;
	memcpy(nnom_output_data_f32, out_f32, sizeof(float32_t) * in_size);

	return NN_SUCCESS;
}
