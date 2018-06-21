/*
 * This is use of cmsis 1D convolution
 *  for reference only
 *  here the bias is not added, which could further
 *  increase the runtime
 */

#include "arm_math.h"
#include "NNFunctions.h"
#include "NNSupportFunctions.h"

void convolve_cmsis_CHW (
                   const q7_t * Im_in,        // * input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q7_t * wt,           // * kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q7_t * bias,         // * bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q7_t * Im_out,             // * output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,
                   q7_t * bufferB
) {
  /* this function is implemented with 
  arm_conv_fast_opt_q15 (
    q15_t * pSrcA,
    uint32_t srcALen,
    q15_t * pSrcB,
    uint32_t srcBLen,
    q15_t * pDst,
    q15_t * pScratch1,
    q15_t * pScratch2)
  */
  uint16_t i, j, k, l, m, n;
  int conv_out;
  char in_row, in_col;
  q15_t outBuffer[dim_im_in+dim_kernel-1];
  q15_t scratch1[dim_im_in+2*dim_kernel - 2];
  q15_t scratch2[dim_kernel];

  q15_t * filter_buffer = bufferA;
  q15_t * input_buffer = (q15_t*) bufferB;

  // do the initialization here
  for (i=0;i<ch_im_out*dim_im_out*dim_im_out;i++) {
    Im_out[i] = 0;
  }

  for(i=0;i<ch_im_out;i++) {
    // process each output channel
    for (j=0;j<ch_im_in;j++) {
      // process each input channels, i.e., image
      for (k=0;k<dim_im_out;k++) {
        // for each output rows
        // first setting the output pointer
        q15_t* output = Im_out + i*dim_im_out*dim_im_out + k*dim_im_out;
        for (l=0;l<dim_kernel;l++) {
          // each line is the results of covolution of dim_kernel lines
          // doing the 1-d convolution here
          if (l + k*stride - padding >= 0 && l + k*stride - padding < dim_im_in) {
            const q15_t* filter_line = wt + (i*dim_kernel*dim_kernel*ch_im_in  // i-th output channel
                               + j*dim_kernel*dim_kernel // j-th input channel
                               + l*dim_kernel); // l-th line

            arm_q7_to_q15_no_shift(filter_line, filter_buffer, dim_kernel);

            const q15_t* input_line = Im_in + (j*dim_im_in*dim_im_in // j-th input channel
                              + (k*stride-padding + l)*dim_im_in);  // startig line equals -1*padding + k*stride + l
 
            arm_q7_to_q15_no_shift(input_line, input_buffer, dim_im_in);

            arm_conv_fast_opt_q15 ((q15_t*)input_buffer, dim_im_in, (q15_t*)filter_buffer, dim_kernel, outBuffer, scratch1, scratch2);
            for (m=0;m<dim_im_out;m++) {
              // outBuffer[dim_im_in+dim_kernel-1]
              // doing some math here to validate the indexing
              // dim_im_out = (dim_im_in + 2*padding - dim_kernel + 1) / stride
              // dim_im_out*stride = dim_im_in + 2*padding - dim_kernel + 1

              // 2*dim_kernel - 2*padding + m*stride - 2 = dim_im_in + dim_kernel - 1
              // m*stride = dim_im_in + 2*padding - dim_kernel + 1
              // matched!
              output[m] += outBuffer[(dim_kernel - padding) + m*stride - 1];
            }
          }
        }
      }
    }
  }
}


