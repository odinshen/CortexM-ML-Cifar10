#ifndef __CNN_H__
#define __CNN_H__

#include "cnn_parameter.h"
#include "cnn_weights.h"
#include "arm_math.h"
#include "NNFunctions.h"

class CNN
{

public:

  typedef struct __CNNOutput
  {
    q7_t *confidence_vector;
    uint8_t label;
    uint32_t execution_time_ms;
  } CNNOutput_t;

  CNN(q7_t* scratch, q7_t* col);
  ~CNN();

  CNNOutput_t classify(uint8_t * image_data);

private:

  q7_t * col_buffer;
  q7_t * img_buffer1;
  q7_t * img_buffer2;

  // static data member here
  static q7_t const conv1_wt[CONV1_IM_CH * CONV1_KER_DIM * CONV1_KER_DIM * CONV1_OUT_CH];
  static q7_t const conv1_bias[CONV1_OUT_CH];
  static q7_t const conv2_wt[CONV2_IM_CH * CONV2_KER_DIM * CONV2_KER_DIM * CONV2_OUT_CH];
  static q7_t const conv2_bias[CONV2_OUT_CH];
  static q7_t const conv3_wt[CONV3_IM_CH * CONV3_KER_DIM * CONV3_KER_DIM * CONV3_OUT_CH];
  static q7_t const conv3_bias[CONV3_OUT_CH];
  static q7_t const ip1_wt[IP1_DIM * IP1_OUT];
  static q7_t const ip1_bias[IP1_OUT];

};

#endif
