#include "cnn.h"
#include <stm32f7xx_hal.h>
#include <arm_nnfunctions.h>
#define IMAGE_SCALE 2

#define CMSIS-NN 1

extern const void* cnn_ts_bias_start;

CNN::CNN(q7_t* scratch, q7_t* col) {
  img_buffer1 = scratch;
  img_buffer2 = img_buffer1 + 32*32*32;
  col_buffer = col;
}

CNN::~CNN() {
}

CNN::CNNOutput_t CNN::classify(uint8_t * image_data) {
  CNNOutput_t cnn_output;
  q7_t output_data[10];
  uint32_t cnn_start_time, cnn_end_time;
  uint8_t *image_mean_ptr = (uint8_t *)(&cnn_ts_bias_start);

  cnn_start_time = HAL_GetTick();

  for (int i=0;i<32*32*3;i++) {
    // normalise current image by subtracting mean of whole dataset
    // saturate value to bit 8 - range ---> -128 <= img_buffer2[i] <= 127
    img_buffer2[i] = (q7_t)__SSAT( ((int)image_data[i] - *image_mean_ptr), 8);
    image_mean_ptr++;
  }
  
  // ------------------------------------------------------------------------------------------------------------
  // LAYER 1
  // ------------------------------------------------------------------------------------------------------------
#ifdef CMSIS-NN
  // CONV 1
/*
  arm_convolve_HWC_q7_RGB(
    img_buffer2, 
    CONV1_IM_DIM, CONV1_IM_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, 
    img_buffer1, CONV1_OUT_DIM, 
    (q15_t *) col_buffer, NULL
  );  
*/
  convolve_HWC_q7_RGB(
    img_buffer2,
    CONV1_IM_DIM, CONV1_IM_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, 
    img_buffer1, CONV1_OUT_DIM, (q15_t*)col_buffer, NULL
  );
  // POOL 1
  arm_maxpool_q7_HWC(
    img_buffer1, 
    CONV1_OUT_DIM, CONV1_OUT_CH, POOL1_KER_DIM, POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, 
    col_buffer, 
    img_buffer2
  );
  // ACTIVATION 1
  arm_relu_q7(
    img_buffer2, 
    POOL1_OUT_DIM * POOL1_OUT_DIM * CONV1_OUT_CH
  );
#else
  // CONV 1
  convolve_HWC_q7_RGB(
    img_buffer2, 
    CONV1_IM_DIM, CONV1_IM_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, 
    img_buffer1, CONV1_OUT_DIM, (q15_t*)col_buffer, NULL
  );
  arm_convolve_HWC_q7_RGB(
    img_buffer2, 
    CONV1_IM_DIM, CONV1_IM_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, 
    img_buffer1, CONV1_OUT_DIM, (q15_t *) col_buffer, NULL
  );  
  // POOL 1
  maxpool_opt_q7_HWC(
    img_buffer1, 
    CONV1_OUT_DIM, CONV1_OUT_CH, POOL1_KER_DIM, POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, 
    col_buffer, img_buffer2
  );
  // ACTIVATION 1
  relu_q7(
    img_buffer2, 
    POOL1_OUT_DIM * POOL1_OUT_DIM * CONV1_OUT_CH
  );
#endif
  

  // ------------------------------------------------------------------------------------------------------------
  // LAYER 2
  // ------------------------------------------------------------------------------------------------------------

#ifdef CMSIS-NN
  // CONV 2
/*
  arm_convolve_HWC_q7_fast(
    img_buffer2, 
    CONV2_IM_DIM, CONV2_IM_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM, CONV2_PADDING, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, 
    img_buffer1, CONV2_OUT_DIM, 
    (q15_t *) col_buffer, 
    NULL
  );
*/
  convolve_HWC_q7_full(
    img_buffer2, 
    CONV2_IM_DIM, CONV2_IM_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM, CONV2_PADDING, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, 
    img_buffer1, CONV2_OUT_DIM, 
    (q15_t*)col_buffer, NULL
  );
  // ACTIVATION 2
  arm_relu_q7(
    img_buffer1, 
    CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH
  );
  // POOL 2
  arm_avepool_q7_HWC(
    img_buffer1, 
    CONV2_OUT_DIM, CONV2_OUT_CH, POOL2_KER_DIM, POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, 
    col_buffer, 
    img_buffer2
  );
#else
  // CONV 2
  convolve_HWC_q7_full(
    img_buffer2, 
    CONV2_IM_DIM, CONV2_IM_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM, CONV2_PADDING, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, 
    img_buffer1, CONV2_OUT_DIM, 
    (q15_t*)col_buffer, NULL
  );
  // ACTIVATION 2
  relu_q7(
    img_buffer1, 
    CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH
  );
  // POOL 2
  avepool_opt_q7_HWC(
    img_buffer1, 
    CONV2_OUT_DIM, CONV2_OUT_CH, POOL2_KER_DIM, POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, 
    col_buffer, 
    img_buffer2
  );
#endif
  
   
  // ------------------------------------------------------------------------------------------------------------
  // LAYER 3
  // ------------------------------------------------------------------------------------------------------------
#ifdef CMSIS-NN
  // CONV3
/*
  arm_convolve_HWC_q7_fast(
    img_buffer2, 
    CONV3_IM_DIM, CONV3_IM_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM, CONV3_PADDING, CONV3_STRIDE, conv3_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, 
    img_buffer1, CONV3_OUT_DIM, 
    (q15_t *) col_buffer, NULL
  );
*/
  convolve_HWC_q7_full(
    img_buffer2, 
    CONV3_IM_DIM, CONV3_IM_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM, CONV3_PADDING, CONV3_STRIDE, conv3_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, 
    img_buffer1, CONV3_OUT_DIM, 
    (q15_t*)col_buffer, NULL
  );

  // ACTIVATION 3
  arm_relu_q7(
    img_buffer1, 
    CONV3_OUT_DIM * CONV3_OUT_DIM * CONV3_OUT_CH
  );
  // POOL 3
  arm_avepool_q7_HWC(
    img_buffer1, 
    CONV3_OUT_DIM, CONV3_OUT_CH, POOL3_KER_DIM, POOL3_PADDING, POOL3_STRIDE, POOL3_OUT_DIM, 
    col_buffer, 
    img_buffer2
  );
#else
  // CONV3
  convolve_HWC_q7_full(
    img_buffer2, 
    CONV3_IM_DIM, CONV3_IM_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM, CONV3_PADDING, CONV3_STRIDE, conv3_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, 
    img_buffer1, CONV3_OUT_DIM, 
    (q15_t*)col_buffer, NULL
  );
  // ACTIVATION 3
  relu_q7(
    img_buffer1, 
    CONV3_OUT_DIM * CONV3_OUT_DIM * CONV3_OUT_CH
  );
  // POOL 3
  avepool_opt_q7_HWC(
    img_buffer1, 
    CONV3_OUT_DIM, CONV3_OUT_CH, POOL3_KER_DIM, POOL3_PADDING, POOL3_STRIDE, POOL3_OUT_DIM, 
    col_buffer, 
    img_buffer2
  );
#endif

  // ------------------------------------------------------------------------------------------------------------
  // LAYER 4
  // ------------------------------------------------------------------------------------------------------------
  // FULLY CONNECTED LAYER
#ifdef CMSIS-NN
  arm_fully_connected_q7_opt(
    img_buffer2, 
    ip1_wt, IP1_DIM, IP1_OUT, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, ip1_bias, 
    output_data, 
    (q15_t *) col_buffer
  );
#else
  fully_connected_q7_x4(
    img_buffer2, 
    ip1_wt, IP1_DIM, IP1_OUT, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, ip1_bias, 
    output_data, 
    (q15_t*)col_buffer
  );
#endif

  // ------------------------------------------------------------------------------------------------------------
  // LAYER 5
  // ------------------------------------------------------------------------------------------------------------
#ifdef CMSIS-NN
  arm_softmax_q7(
    output_data, 
    10, 
    img_buffer1
  );
#else
  // SOFTMAX 
  softmax_q7(
    output_data, 
    10, 
    img_buffer1
  ); // output of softmax in img_buffer1 is not used
#endif

  cnn_end_time = HAL_GetTick();

  int max_value = -128;
  int max_id = -1;

  for (int i=0;i<10;i++) {
    if (output_data[i] > max_value) {
      max_value = output_data[i];
      max_id = i;
    }
  }

  cnn_output.confidence_vector = img_buffer1;
  cnn_output.label = max_id;
  cnn_output.execution_time_ms = (cnn_end_time - cnn_start_time);

  return cnn_output;

}

const q7_t CNN::conv1_wt[CONV1_IM_CH*CONV1_KER_DIM*CONV1_KER_DIM*CONV1_OUT_CH] = CNN_CONV1_WT;
const q7_t CNN::conv1_bias[CONV1_OUT_CH] = CNN_CONV1_BIAS;
const q7_t CNN::conv2_wt[CONV2_IM_CH*CONV2_KER_DIM*CONV2_KER_DIM*CONV2_OUT_CH] = CNN_CONV2_WT;
const q7_t CNN::conv2_bias[CONV2_OUT_CH] = CNN_CONV2_BIAS;
const q7_t CNN::conv3_wt[CONV3_IM_CH*CONV3_KER_DIM*CONV3_KER_DIM*CONV3_OUT_CH] = CNN_CONV3_WT;
const q7_t CNN::conv3_bias[CONV3_OUT_CH] = CNN_CONV3_BIAS;
const q7_t CNN::ip1_wt[IP1_DIM*IP1_OUT] = CNN_IP1_WT;
const q7_t CNN::ip1_bias[IP1_OUT] = CNN_IP1_BIAS;
