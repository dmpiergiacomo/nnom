#include "nnom.h"

/* Weights, bias and Q format */


/* output q format for each layer */
#define LAMBDA_OUTPUT_DEC 7
#define LAMBDA_OUTPUT_OFFSET 0

/* bias shift and output shift for none-weighted layer */

/* tensors and configurations for each layer */
static int8_t nnom_input_data[4] = {0};

const nnom_shape_data_t tensor_input_1_dim[] = {4, 1};
const nnom_qformat_param_t tensor_input_1_dec[] = {7};
const nnom_qformat_param_t tensor_input_1_offset[] = {0};
const nnom_tensor_t tensor_input_1 = {
    .p_data = (void*)nnom_input_data,
    .dim = (nnom_shape_data_t*)tensor_input_1_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_input_1_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_input_1_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};

const nnom_io_config_t input_1_config = {
    .super = {.name = "input_1"},
    .tensor = (nnom_tensor_t*)&tensor_input_1
};

static float32_t nnom_output_data_f32[4] = {0.};
const nnom_lambda_config_t lambda_config = {
    .super = {.name = "lambda"},
    .run_func_name = l2norm_run,
    .build_func_name = l2norm_build,
    .free_func_name = l2norm_free,
    .parameters = nnom_output_data_f32
};
static int8_t nnom_output_data[4] = {0};

const nnom_shape_data_t tensor_output0_dim[] = {4};
const nnom_qformat_param_t tensor_output0_dec[] = {LAMBDA_OUTPUT_DEC};
const nnom_qformat_param_t tensor_output0_offset[] = {0};
const nnom_tensor_t tensor_output0 = {
    .p_data = (void*)nnom_output_data,
    .dim = (nnom_shape_data_t*)tensor_output0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_output0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_output0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_io_config_t output0_config = {
    .super = {.name = "output0"},
    .tensor = (nnom_tensor_t*)&tensor_output0
};
/* model version */
#define NNOM_MODEL_VERSION (10000*0 + 100*4 + 3)

/* nnom model */
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[3];

	check_model_version(NNOM_MODEL_VERSION);
	new_model(&model);

	layer[0] = input_s(&input_1_config);
	layer[1] = model.hook(lambda_s(&lambda_config), layer[0]);
	layer[2] = model.hook(output_s(&output0_config), layer[1]);
	model_compile(&model, layer[0], layer[2]);
	return &model;
}
