/* Copyright 2019 Sipeed Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "kpu.h"

/*
//层参数列表，共16层kpu_layer_argument_t la[] __attribute__((aligned(128))) = {
// 第0层{
 .kernel_offset.data = {
  .coef_row_offset = 0,		//固定为0
  .coef_column_offset = 0	//固定为0
 },
 .image_addr.data = {
//图像输入输出地址，一个在前，一个在后，下一层运算的时候翻过来，可以避免拷贝工作。
  .image_dst_addr = (uint64_t)0x6980,	//图像输出地址，int((0 if idx & 1 else
(img_ram_size - img_output_size)) / 64) .image_src_addr = (uint64_t)0x0
//图像加载地址
 },
 .kernel_calc_type_cfg.data = {
  .load_act = 1,
//使能激活函数，必须使能（硬件设计如此），不使能则输出全为0 .active_addr = 0,
//激活参数加载首地址，在kpu_task_init里初始化为激活折线表 .row_switch_addr =
0x5,	//图像宽占用的单元数，一个单元64Byte.  ceil(width/64)=ceil(320/64)=5
  .channel_switch_addr = 0x4b0,			//单通道占用的单元数.
row_switch_addr*height=5*240=1200=0x4b0 .coef_size = 0, //固定为0 .coef_group =
1			//一次可以计算的组数，因为一个单元64字节，
                                                        //所以宽度>32，设置为1；宽度17~32，设置为2；宽度<=16，设置为4
 },
 .interrupt_enabe.data = {
  .depth_wise_layer = 0,	//常规卷积层,设置为0
  .ram_flag = 0,			//固定为0
  .int_en = 0,				//失能中断
  .full_add = 0				//固定为0
 },
 .dma_parameter.data = {	//DMA传输参数
  .dma_total_byte = 307199,		//该层输出16通道，即 19200*16=308200
  .send_data_out = 0,			//使能输出数据
  .channel_byte_num = 19199		//输出单通道的字节数，因为后面是2x2
pooling, 所以大小为160*120=19200
 },
 .conv_value.data = {		//卷积参数，y = (x*arg_x)>>shr_x
  .arg_x = 0x809179,		//24bit	乘法参数
  .arg_w = 0x0,
  .shr_x = 8,				//4bit	移位参数
  .shr_w = 0
 },
 .conv_value2.data = {		//arg_add = kernel_size * kernel_size *
bw_div_sw * bx_div_sx =3x3x?x? .arg_add = 0
 },
 .write_back_cfg.data = {	//写回配置
  .wb_row_switch_addr = 0x3,		//ceil(160/64)=3
  .wb_channel_switch_addr = 0x168,	//120*3=360=0x168
  .wb_group = 1						//输入行宽>32,设置为1
 },
 .image_size.data = {	//输入320*240，输出160*120
  .o_col_high = 0x77,
  .i_col_high = 0xef,
  .i_row_wid = 0x13f,
  .o_row_wid = 0x9f
 },
 .kernel_pool_type_cfg.data = {
  .bypass_conv = 0,		//硬件不能跳过卷积，固定为0
  .pad_value = 0x0,		//边界填充0
  .load_para = 1,		//硬件不能跳过归一化，固定为1
  .pad_type = 0,		//使用填充值
  .kernel_type = 1,		//3x3设置为1， 1x1设置为0
  .pool_type = 1,		//池化类型，步长为2的2x2 max pooling
  .dma_burst_size = 15,	//dma突发传送大小，16字节；脚本中固定为16
  .bwsx_base_addr = 0,	//批归一化首地址，在kpu_task_init中初始化
  .first_stride = 0		//图像高度不超过255；图像高度最大为512。
 },
 .image_channel_num.data = {
  .o_ch_num_coef = 0xf,
//一次性参数加载可计算的通道数，16通道。4K/单通道卷积核数
                                                //o_ch_num_coef =
math.floor(weight_buffer_size / o_ch_weights_size_pad) .i_ch_num = 0x2,
//输入通道，3通道 RGB .o_ch_num = 0xf		//输出通道，16通道
 },
 .kernel_load_cfg.data = {
  .load_time = 0,		//卷积加载次数，不超过72KB，只加载一次
  .para_size = 864,		//卷积参数大小864字节，864=3(RGB)*9(3x3)*2*16
  .para_start_addr = 0,	//起始地址
  .load_coor = 1		//允许加载卷积参数
 }
},
   //第0层参数结束……
};

*/

//激活函数折点表，设置为y=x，即直接输出卷积结果
// y=(uint8_t)((((uint64_t)(x - x_start) * y_mul) >> shift) + bias);

kpu_activate_table_t active_addr __attribute__((aligned(256))) =
    {.activate_para =
         {// x =36bit
          {.data = {.shift_number = 0, .y_mul = 0, .x_start = 0x800000000}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}},
          {.data = {.shift_number = 0, .y_mul = 1, .x_start = 0}}},
     .activate_para_bias0.data = {.result_bias = {0, 0, 0, 0, 0, 0, 0, 0}},
     .activate_para_bias1.data = {.result_bias = {0, 0, 0, 0, 0, 0, 0, 0}}};

// y = (x*norm_mul)>>norm_shift + norm_add
kpu_batchnorm_argument_t bwsx_base_addr[1024] __attribute__((aligned(128))) = {
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
    {.batchnorm.data = {.norm_mul = 1, .norm_add = 0x0, .norm_shift = 0}},
};

//卷积参数
kpu_layer_argument_t la __attribute__((aligned(128)));
// max for 3in*3out, you can modify it
uint16_t conv_data_u16[9 * 3 * 3] __attribute__((aligned(128)));

//池化类型，0表示跳过
// 0x1 代表步长为 2 的 2x2 max pooling,
// 0x2 代表步长为 2 的 2x2 mean pooling,
// 0x3 代表步长为 4 的 4x4 max pooling,
// 0x4 代表步长为 4 的 4x4 mean pooling,
// 0x5 代表步长为 2 的 2x2 left_top pooling,
// 0x6 代表步长为 2 的 2x2 right_bottom pooling,
// 0x7 代表步长为 4 的 4x4 left_top pooling,
// 0x8 代表步长为 1 的 2x2 mean pooling,
// 0x9 代表步长为 1 的 2x2 max pooling
#define AI_MEM_SIZE 0x200000

static float min(float *data, uint32_t len) {
  int i;
  float m = data[0];
  for (i = 0; i < len; i++) {
    if (data[i] < m) m = data[i];
  }
  return m;
}

static float max(float *data, uint32_t len) {
  int i;
  float m = data[0];
  for (i = 0; i < len; i++) {
    if (data[i] > m) m = data[i];
  }
  return m;
}

// global var: la, active_addr, bwsx_base_addr
static void conv_float2u16(float *data, uint16_t *data_u16, int len) {
  float dmin, drange, scale, arg_x;
  uint16_t bias, y_mul;
  int i, shift_number;
  dmin = min(data, len);
  drange = max(data, len) - dmin;
  scale = (65535.0 / drange);
  // scale conv
  printf("convert conv parm: -------------\r\n");
  for (i = 0; i < len; i++) {
    data_u16[i] = (uint16_t)((data[i] - dmin) * scale);
    printf("0x%04x\t", data_u16[i]);
    if (i % 9 == 8) printf("\r\n");
  }
  // set arg_x & shr_x
  printf("set arg_x & shr_x: -------------\r\n");
  arg_x = scale * (dmin >= 0 ? dmin : -dmin);
  for (i = 0; (arg_x < (float)(0x400000)) && (arg_x != 0); i++) {
    arg_x *= 2;
    // printf("argx=%f, shrx=%d\r\n", arg_x, i);
  }
  la.conv_value.data.arg_x =
      dmin >= 0 ? (uint32_t)(arg_x) : (uint32_t)(0x1000000 - (uint32_t)arg_x);
  la.conv_value.data.shr_x = i;
  printf("arg_x=0x%x, shr_x=%d\r\n", la.conv_value.data.arg_x,
         la.conv_value.data.shr_x);
  // set act table
  printf("set act table: -------------\r\n");
  printf("origin scale=%f\r\n", scale);
  scale = 1.0 / scale;
  for (i = 0; scale <= 16383.0; i++) {
    scale = scale * 2;
  }
  shift_number = i;
  y_mul = (uint16_t)(scale);
  printf("shift_number=%d, y_mul=%d\r\n", shift_number, y_mul);
  for (i = 1; i < 16; i++) {
    active_addr.activate_para[i].data.shift_number = shift_number;
    active_addr.activate_para[i].data.y_mul = y_mul;
    active_addr.activate_para[i].data.x_start = 0;
  }
  return;
}

void layer_conv_init(kpu_task_t *task, uint16_t w, uint16_t h, uint8_t ch_in, uint8_t ch_out, int8_t *weights, int8_t w_shift)
{
  	int tmp;

	// conv_float2u16(conv_data, conv_data_u16, 9 * ch_in * ch_out); //3x3 kernel

	la.kernel_offset.data.coef_row_offset = 0;     //固定为0
	la.kernel_offset.data.coef_column_offset = 0;  //固定为0
	//激活函数配置-
	la.kernel_calc_type_cfg.data.load_act = 1;  //使能激活函数
	la.kernel_calc_type_cfg.data.active_addr = (uint64_t)&active_addr;
	//初始化激活表
	// row_switch_addr = math.ceil(i_row_wid / 64)
	// channel_switch_addr = i_col_high * row_switch_addr
	la.kernel_calc_type_cfg.data.row_switch_addr =
		(w + 63) / 64;  //图像宽度占用的单元数
	la.kernel_calc_type_cfg.data.channel_switch_addr = (w + 63) / 64 * h;
	la.kernel_calc_type_cfg.data.coef_size = 0;  //固定为0
	la.kernel_calc_type_cfg.data.coef_group = 1;

	//中断设置--
	la.interrupt_enabe.data.depth_wise_layer = 0;  //常规卷积层
	la.interrupt_enabe.data.int_en = 1;            //使能中断
	la.interrupt_enabe.data.full_add = 0;          //??
	la.interrupt_enabe.data.ram_flag = 1;          //??
	// dma设置，知道是输出数据使用的DMA--
	la.dma_parameter.data.dma_total_byte =
		w * h * ch_out - 1;                   //总共的DMA传输数量
	la.dma_parameter.data.send_data_out = 1;  //使能数据的dma输出
	la.dma_parameter.data.channel_byte_num = w * h - 1;  //单通道的DMA传输数量
	//卷积运算参数设置--
	// arg_x 为24bit,shr_x 为4bit, 在conv_float2u16中设置
	/*
	la.conv_value.data.arg_x = 0;
	la.conv_value.data.shr_x = 0;
	la.conv_value.data.arg_w = 0;
	la.conv_value.data.shr_w = 0;
	la.conv_value2.data.arg_add = 0;
	*/
	//写回设置--
	la.write_back_cfg.data.wb_row_switch_addr = (w + 63) / 64;  // ceil(16/64)=1
	la.write_back_cfg.data.wb_channel_switch_addr = (w + 63) / 64 * h;  // 16*1
	la.write_back_cfg.data.wb_group = 1;                                // 64/w
	//图像尺寸设置--
	la.image_size.data.i_row_wid = w - 1;  //输入长宽
	la.image_size.data.i_col_high = h - 1;
	la.image_size.data.o_row_wid = w - 1;  //输出长宽
	la.image_size.data.o_col_high = h - 1;
	//池化类型设置-
	la.kernel_pool_type_cfg.data.bypass_conv = 0;  //不略过卷积
	la.kernel_pool_type_cfg.data.pad_value = 0x0;  //边界填充0
	la.kernel_pool_type_cfg.data.load_para = 1;    //允许归一化
	la.kernel_pool_type_cfg.data.pad_type = 0;     //使用填充值
	la.kernel_pool_type_cfg.data.kernel_type = 1;  // 3x3
	la.kernel_pool_type_cfg.data.pool_type = 0;    //池化类型，跳过
	la.kernel_pool_type_cfg.data.dma_burst_size = 15;  // dma突发传送大小，16字节
	la.kernel_pool_type_cfg.data.bwsx_base_addr = (uint64_t)&bwsx_base_addr;
	//批归一化首地址
	la.kernel_pool_type_cfg.data.first_stride =
		h < 256 ? 0 : 1;  //图像高度未超过255
	//图像通道设置--
	la.image_channel_num.data.o_ch_num_coef =
		ch_out - 1;  //一次性参数加载可计算的通道数
	la.image_channel_num.data.i_ch_num = ch_in - 1;   //输入通道
	la.image_channel_num.data.o_ch_num = ch_out - 1;  //输出通道
	//卷积参数设置-
	la.kernel_load_cfg.data.load_time = 0;  //卷积加载次数，不超过72KB，只加载一次
	la.kernel_load_cfg.data.para_size = 2 * 9 * ch_in * ch_out;  //卷积参数大小
	la.kernel_load_cfg.data.para_start_addr = (uint64_t)conv_data_u16;
	//起始地址
	la.kernel_load_cfg.data.load_coor = 1;  //允许加载卷积参数
	//计算地址设置--
	la.image_addr.data.image_src_addr = (uint64_t)0x0;  //一个为0
	la.image_addr.data.image_dst_addr =
		(uint64_t)(AI_MEM_SIZE / 64 - (w + 63) / 64 * h * ch_out);

	/* init kpu task*/
	task->layers = &la;
	task->layers_length = 1;   //单层
	task->eight_bit_mode = 1;  // 16bit模式
	task->output_scale = 1.0;  //输出的缩放
	task->output_bias = 0;     //输出的偏置
	return;
}

void layer_conv_run(kpu_task_t *task, uint8_t *img_src, uint8_t *img_dst,
                    plic_irq_callback_t callback) {
	/* start to calculate */
	kpu_run(task, DMAC_CHANNEL5, img_src, img_dst, callback);
	return;
}

#include "nnom.h"

// NNoM should has fuse the batch normalization to the previous convolution
// However, convolution in KPU doesnt add the bias, so we use kpu's
// normalization to add the bias
void set_bias_to_normalization(kpu_batchnorm_argument_t *bn, int8_t *bias, int8_t b_shift, uint32_t len)
{
  	for (i = 0; i < len; i++)
	{
		bn[i].batchnorm.data.norm_mul = 1;
		bn[i].batchnorm.data.norm_add = bias[i];
		bn[i].batchnorm.data.norm_shift = shift;
  	}
}

uin8_t flag_done = 0;
int conv_done_cb(void *ctx)
{
	flag_done = 1;
}

// it must meet all conditions in KPU
void nnom_kpu_conv(nnom_layer_t *layer)
{
	nnom_conv2d_layer_t *cl = (nnom_conv2d_layer_t *)layer;
	kpu_task_t task;

	// preperation
	set_bias_to_normalization(bwsx_base_addr, cl->bias->p_value, cl->bias_shift, layer->out->shape.c);
	// copy weights, size 2 * 9 * ch_in * ch_out
	int size = cl->kernel.w * cl->kernel.h * layer->in->shape.c * layer->out->shape.c;
	memcpy(conv_data, cl->weight->p_value, size);

	// Todo, must check the conditions, kernels, shapes, ...
	layer_conv_init(&task, layer->in->shape.w, layer->in->shape.h,
					layer->in->shape.c, layer->out->shape.c, conv_data, cl->output_shift);

	// run, do i need to copy the data into the AI buffer? or the api has done the job?
	flag_done = 0;
	layer_conv_run(&task, layer->in->mem->blk, layer->out->mem->blk, conv_done_cb);

	// not sure if this works
	while(1)
	{
		if(flag_done)
			break;
	}
	return;
}
