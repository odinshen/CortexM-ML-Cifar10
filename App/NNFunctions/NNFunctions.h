#ifndef _NNFunctions_H_
#define _NNFunctions_H_

#include "NNSupportFunctions.h"
#include "NNCommonTable.h"

/*
 * Types
 */

typedef enum {INTER_CHANNEL, INTRA_CHANNEL} norm_type;
typedef enum {AVERAGE, MAX} pool_type;

#define ACTIVATION_Q7(in) in>0 ? (q7_t)__SSAT((in>>7), 8) : 0

/*
 *  Convolution Functions
 *
 *  Here are different varieties of the convolution implementation.
 *  convolve_{input_format}_{precision}_{special description}
 *  Input format: HWC and CHW
 *  Precision: q7 (1 byte) or q15 (2 bytes)
 *  Special description
 */

/*
 * The basic function is the universal one that works with
 * different input/output cases
 *
 */

#ifdef __cplusplus
extern "C"
{
#endif


arm_status convolve_HWC_q7_basic(
                   const q7_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q7_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q7_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q7_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
);

arm_status convolve_HWC_q15_basic(
                   const q15_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q15_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q15_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q15_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
);

arm_status convolve_CHW_q15_basic(
                   const q15_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q15_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q15_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q15_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
);


/* This function is the version with full list of optimization tricks, but with
 * some contraints:
 *   ch_im_in is multiple of 4
 *   ch_im_out is multiple of 2
 *   dim_im_out is multiple of 2
 */

arm_status convolve_HWC_q7_full(
                   const q7_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q7_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q7_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q7_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
);

/*
 * This function is the q15 version with 2x2 mat_mult kernel
 * ch_im_oout must be multiple of 2
 * dim_im_in must be multiple of 2
 */

arm_status convolve_HWC_q15_full(
                   const q15_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q15_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q15_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q15_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
);

arm_status convolve_HWC_q15_full_x4(
                   const q15_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q15_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q15_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q15_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
);

arm_status convolve_HWC_q15_full_x4_ref(
                   const q15_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q15_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q15_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q15_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
);

/*
 * The basic function is the specific function that designed for handling
 * the first convolution layer, i.e., with ch_im_in of 3
 */

arm_status convolve_HWC_q7_RGB(
                   const q7_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q7_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q7_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q7_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
);

void convolve_cmsis_CHW (
                   const q7_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q7_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q7_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q7_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
);


arm_status convolve_HWC_q15_RGB(
                   const q15_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q15_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q15_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q15_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
);

arm_status separable_conv_HWC_q7_ref (
                   const q7_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q7_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q7_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q7_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
);

arm_status separable_conv_HWC_q7 (
                   const q7_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q7_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q7_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q7_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
);

/*
 * The fully connected layer
 *
 *
 */

arm_status fully_connected_q7(
            const q7_t * pV,     // pointer to vector
            const q7_t * pM,     // pointer to matrix
            const uint16_t dim_vec, // numRow of
            const uint16_t num_of_rows, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,
            q7_t * pOut,        // output operand
            q15_t * vec_buffer
);

arm_status fully_connected_q7_x2(
            const q7_t * pV,     // pointer to vector
            const q7_t * pM,     // pointer to matrix
            const uint16_t dim_vec, // length of the vector
            const uint16_t num_of_rows, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,
            q7_t * pOut,        // output operand
            q15_t * vec_buffer
);

arm_status fully_connected_q15(
            const q15_t * pV,     // pointer to vector
            const q15_t * pM,     // pointer to matrix
            const uint16_t dim_vec, // numRow of
            const uint16_t num_of_rows, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q15_t * bias,
            q15_t * pOut,        // output operand
            q15_t * vec_buffer
);

arm_status fully_connected_q7_x4(
            const q7_t * pV,     // pointer to vector
            const q7_t * pM,     // pointer to matrix
            const uint16_t dim_vec, // length of the vector
            const uint16_t num_of_rows, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,
            q7_t * pOut,        // output operand
            q15_t * vec_buffer
);

arm_status fully_connected_q7_x4_ref(
            const q7_t * pV,     // pointer to vector
            const q7_t * pM,     // pointer to matrix
            const uint16_t dim_vec, // length of the vector
            const uint16_t num_of_rows, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,
            q7_t * pOut,        // output operand
            q15_t * vec_buffer
);

arm_status fully_connected_q15_x4(
            const q15_t * pV,     // pointer to vector
            const q15_t * pM,     // pointer to matrix
            const uint16_t dim_vec, // length of the vector
            const uint16_t num_of_rows, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q15_t * bias,
            q15_t * pOut,        // output operand
            q15_t * vec_buffer
);

arm_status fully_connected_q7_q15_x4(
            const q15_t * pV,     // pointer to vector
            const q7_t * pM,     // pointer to matrix
            const uint16_t dim_vec, // length of the vector
            const uint16_t num_of_rows, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,
            q15_t * pOut,        // output operand
            q15_t * vec_buffer
);


#ifdef __cplusplus
}
#endif



/*
 *  Some utility functions that handle the main computation
 *  Derived based on CMSIS_DSP with merged features for NN
 *  operations.
 */

#ifdef __cplusplus
extern "C"
{
#endif

// matrix multiplication with bias built in
q7_t* mat_mult_kernel_q7_q15(
            const q7_t * pA,     // pointer to operand A
            const q15_t * pInBuffer,     // pointer to operand B, always conssists of 2 vectors
            const uint16_t ch_im_out, // numRow of A
            const uint16_t numCol_A, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,   // the bias 
            q7_t * pOut        // output operand
);

q7_t* mat_mult_kernel_q7_q15_shuffle(
            const q7_t * pA,     // pointer to operand A
            const q15_t * pInBuffer,     // pointer to operand B, always conssists of 2 vectors
            const uint16_t ch_im_out, // numRow of A
            const uint16_t numCol_A, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,   // the bias 
            q7_t * pOut        // output operand
);

// matrix multiplication with bias and relu built in
q7_t* mat_mult_relu_kernel_q7_q15(
            const q7_t * pA,     // pointer to operand A
            const q15_t * pInBuffer,     // pointer to operand B, always conssists of 2 vectors
            const uint16_t ch_im_out, // numRow of A
            const uint16_t numCol_A, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,   // the bias 
            q7_t * pOut        // output operand
);

q7_t* mat_mult_kernel_activation_q7_q15(
            const q7_t * pA,     // pointer to operand A
            const q15_t * pInBuffer,     // pointer to operand B, always conssists of 2 vectors
            const uint16_t ch_im_out, // numRow of A
            const uint16_t numCol_A, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,   // the bias 
            q7_t * pOut        // output operand
);


#ifdef __cplusplus
}
#endif


/*
 *  Other functions
 *  These layers are typically not timing critical
 *  Basic implementation is supported here
 */

#ifdef __cplusplus
extern "C"
{
#endif

void relu_q7(
        q7_t * data,
        uint16_t size
);

void relu_q7_ref(
        q7_t * data,
        uint16_t size
);

void relu_simd_q7(
        q7_t * data,
        uint16_t size
);

void relu_q15(
        q15_t * data,
        uint16_t size
);

void sigmoid_direct_q7(
        q7_t * data,            // pointer to the data array
        uint16_t size,          // size of the pointer array
        uint16_t int_width      // bit-width of the integer part, assume to be smaller than 3
);

void tanh_direct_q7(
        q7_t * data,            // pointer to the data array
        uint16_t size,          // size of the pointer array
        uint16_t int_width      // bit-width of the integer part, assume to be smaller than 3
); 


void sigmoid_direct_q15(
        q15_t * data,            // pointer to the data array
        uint16_t size,          // size of the pointer array
        uint16_t int_width      // bit-width of the integer part, assume to be smaller than 3
); 

void tanh_direct_q15(
        q15_t * data,            // pointer to the data array
        uint16_t size,          // size of the pointer array
        uint16_t int_width      // bit-width of the integer part, assume to be smaller than 3
);

void sigmoid_dual_q7(
        q7_t* data,
        uint16_t size,
	uint16_t int_width
);

void tanh_direct_q7(
        q7_t* data,
        uint16_t size,
	uint16_t int_width
);

void tanh_direct_q15(
        q15_t* data,
        uint16_t size,
	uint16_t int_width
);


void norm_q7_HWC (
        const q7_t * Im_in,          // input image
        const uint16_t dim_im_in,    // input image dimention
        const uint16_t ch_im_in,     // number of input image channels
        const uint16_t window_size,  //
        const norm_type type,     // type of norm operation
        q7_t * Im_out
);

void maxpool_q7_HWC (
        const q7_t * Im_in,         // input image
        const uint16_t dim_im_in,   // input image dimension
        const uint16_t ch_im_in,    // number of input image channels
        const uint16_t dim_kernel,  // window kernel size
        const uint16_t padding,     // padding sizes
        const uint16_t stride,      // stride 
        const uint16_t dim_im_out,  // output image dimension
        q7_t * bufferA,             // a buffer for local storage
        q7_t * Im_out
);

void maxpool_opt_q7_HWC (
        q7_t * Im_in,         // input image, not const, i.e., destructive on this operation
        const uint16_t dim_im_in,   // input image dimension
        const uint16_t ch_im_in,    // number of input image channels
        const uint16_t dim_kernel,  // window kernel size
        const uint16_t padding,     // padding sizes
        const uint16_t stride,      // stride
        const uint16_t dim_im_out,  // output image dimension
        q7_t * bufferA,             // a buffer for local storage
        q7_t * Im_out
);

void avepool_q7_HWC (
        const q7_t * Im_in,         // input image
        const uint16_t dim_im_in,   // input image dimension
        const uint16_t ch_im_in,    // number of input image channels
        const uint16_t dim_kernel,  // window kernel size
        const uint16_t padding,     // padding sizes
        const uint16_t stride,      // stride 
        const uint16_t dim_im_out,  // output image dimension
        q7_t * bufferA,             // a buffer for local storage
        q7_t * Im_out
);

//
void avepool_opt_q7_HWC (
        const q7_t * Im_in,         // input image
        const uint16_t dim_im_in,   // input image dimension
        const uint16_t ch_im_in,    // number of input image channels
        const uint16_t dim_kernel,  // window kernel size
        const uint16_t padding,     // padding sizes
        const uint16_t stride,      // stride 
        const uint16_t dim_im_out,  // output image dimension
        q7_t * bufferA,             // a buffer to store temp accumulator
        q7_t * Im_out
);

void softmax_q7(
        const q7_t* vec_in,        // input vector
        const uint16_t dim_vec,    // dimension of the vector
        q7_t* p_out          // output vector
);

void softmax_q15(
        const q15_t* vec_in,        // input vector
        const uint16_t dim_vec,    // dimension of the vector
        q15_t* p_out          // output vector
);

#ifdef __cplusplus
}
#endif

#endif
