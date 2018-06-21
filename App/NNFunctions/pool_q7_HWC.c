/* ----------------------------------------------------------------------    
* Description:   Q7 version of pool
*
*
*
*
* -------------------------------------------------------------------- */
#include "NNFunctions.h"
#include "NNSupportFunctions.h"

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
) {
  int16_t i_ch_in, i_x, i_y;
  int16_t k_x, k_y;
  
  for(i_ch_in=0;i_ch_in<ch_im_in;i_ch_in++) {
    for(i_y=0;i_y<dim_im_out;i_y++) {
      for(i_x=0;i_x<dim_im_out;i_x++) {
        int sum = 0;
        int count = 0;
        for (k_y = i_y*stride-padding; k_y < i_y*stride-padding+dim_kernel; k_y++) {
          for (k_x = i_x*stride-padding;k_x < i_x*stride-padding+dim_kernel; k_x++) {
            if (k_y >= 0 && k_x >= 0 && k_y<dim_im_in && k_x<dim_im_in) {
              sum += Im_in[i_ch_in + ch_im_in*(k_x+k_y*dim_im_in)];
              count++;
            }
          }
        }
        Im_out[i_ch_in+ch_im_in*(i_x+i_y*dim_im_out)] = sum/count;
      }
    }
  }
}

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
) {
  int16_t i_ch_in, i_x, i_y;
  int16_t k_x, k_y;

  for(i_ch_in=0;i_ch_in<ch_im_in;i_ch_in++) {
    for(i_y=0;i_y<dim_im_out;i_y++) {
      for(i_x=0;i_x<dim_im_out;i_x++) {
        int max = -129;
	// pass through all elements of receptive field
        for (k_y = i_y*stride-padding; k_y < i_y*stride-padding+dim_kernel; k_y++) { 
          for (k_x = i_x*stride-padding; k_x < i_x*stride-padding+dim_kernel; k_x++) {
	    // within the elements of the image (not on the padding as these are all zeroes)
            if (k_y >= 0 && k_x >= 0 && k_y<dim_im_in && k_x<dim_im_in) {
              if (Im_in[i_ch_in + ch_im_in*(k_x+k_y*dim_im_in)] > max) {
                max = Im_in[i_ch_in + ch_im_in*(k_x+k_y*dim_im_in)];
              }
            }
          }
        }
        Im_out[i_ch_in+ch_im_in*(i_x+i_y*dim_im_out)] = max;
      }
    }
  }
}

static void buffer_scale_back_q15_to_q7(
        q15_t * buffer, 
        q7_t * target, 
        uint16_t length, 
        uint16_t scale) {
  for (int i=0;i<length;i++) {
    target[i] = (q7_t) (buffer[i] / scale);
  }
}

static void accumulate_q7_to_q15(
        q15_t * base,              // accumulate base
        q7_t * target,            // 
        const uint16_t length      //
) {
  q15_t* pCnt = base;
  q7_t * pV = target;
  q31_t * v1, v2, vo1, vo2;
  uint16_t cnt = length >> 2;

  while (cnt > 0u) {
    q31_t value = *__SIMD32(pV)++;
    v1 = __SXTB16(__ROR(value, 8));
    v2 = __SXTB16(value);
#ifndef ARM_MATH_BIG_ENDIAN

    vo2 = __PKHTB(v1, v2, 16);
    vo1 = __PKHBT(v2, v1, 16);

#else

    vo1 = __PKHTB(v1, v2, 16);
    vo2 = __PKHBT(v2, v1, 16);

#endif

    q31_t in = *__SIMD32(pCnt);
    *__SIMD32(pCnt)++ = __QADD16(vo1, in);

    in = *__SIMD32(pCnt);
    *__SIMD32(pCnt)++ = __QADD16(vo2, in);
   
    cnt --;
  }
  cnt = length & 0x3;
  while (cnt > 0u) {
    *pCnt++ += *pV++;
    cnt --;
  }
}

static void compare_and_replace_if_larger_q7 (
        q7_t * base,               // base data
        q7_t * target,             // compare target
        const uint16_t length      // data length
) {
  q7_t* pIn = base;
  q7_t* pCom = target;
  union MyWord in; 
  union MyWord com; 
  uint16_t cnt = length >> 2;

  while (cnt > 0u) {
    in.word = *__SIMD32(pIn);
    com.word = *__SIMD32(pCom)++;

    //// conditional assignment version
    //// seems to be slower than if-else version
    //// probably because it has to do assignment both ways
    //in.bytes[0] = (com.bytes[0] > in.bytes[0]) ? com.bytes[0] : in.bytes[0];
    //in.bytes[1] = (com.bytes[1] > in.bytes[1]) ? com.bytes[1] : in.bytes[1];
    //in.bytes[2] = (com.bytes[2] > in.bytes[2]) ? com.bytes[2] : in.bytes[2];
    //in.bytes[3] = (com.bytes[3] > in.bytes[3]) ? com.bytes[3] : in.bytes[3];

    // if version
    if (com.bytes[0] > in.bytes[0])
      in.bytes[0] = com.bytes[0];
    if (com.bytes[1] > in.bytes[1]) 
      in.bytes[1] = com.bytes[1]; 
    if (com.bytes[2] > in.bytes[2]) 
      in.bytes[2] = com.bytes[2]; 
    if (com.bytes[3] > in.bytes[3]) 
      in.bytes[3] = com.bytes[3]; 

    *__SIMD32(pIn)++ = in.word;

    cnt --;
  }

  cnt = length & 0x3;
  while (cnt > 0u) {
    if (*pCom > *pIn) {
      *pIn = *pCom;
    }
    pIn++;
    pCom++;
    cnt --;
  }
}

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
) {
  int16_t i_ch_in, i_x, i_y;
  int16_t k_x, k_y;

  // first does the pooling along x axis
  for(i_y=0;i_y<dim_im_in;i_y++) {

    for(i_x=0;i_x<dim_im_out;i_x++) {
    // for each output pixel
      q7_t* target = Im_in + (i_y*dim_im_in+i_x)*ch_im_in;
      q7_t* win_start;
      q7_t* win_stop;
      if (i_x*stride-padding < 0) {
        win_start = target;
      } else {
        win_start = Im_in + (i_y*dim_im_in + i_x*stride-padding )*ch_im_in;
      }

      if (i_x*stride-padding+dim_kernel >= dim_im_in) {
        win_stop = Im_in + (i_y*dim_im_in + dim_im_in)*ch_im_in;
      } else {
        win_stop = Im_in + (i_y*dim_im_in + i_x*stride-padding+dim_kernel )*ch_im_in;
      }

      // first step is to copy over initial data
      arm_copy_q7(win_start, target, ch_im_in);

      // start the max operation from the second part
      win_start += ch_im_in;
      for ( ;win_start<win_stop;win_start+=ch_im_in) {
        compare_and_replace_if_larger_q7(target, win_start, ch_im_in);
      }
    }
  }

  // then does the pooling along y axis
  for(i_y=0;i_y<dim_im_out;i_y++) {

  // for each output row      
    q7_t* target = Im_out + i_y*dim_im_out*ch_im_in;
    q7_t* row_start;
    q7_t* row_end;
    // setting the starting row
    if (i_y*stride-padding<0) {
      row_start = Im_in;
    } else {
      row_start = Im_in + (i_y*stride-padding)*dim_im_in*ch_im_in;
    }
    // setting the stopping row
    if (i_y*stride-padding+dim_kernel >= dim_im_in) {
      row_end = Im_in + dim_im_in*dim_im_in*ch_im_in;
    } else {
      row_end = Im_in + (i_y*stride-padding+dim_kernel)*dim_im_in*ch_im_in;
    }

    // copy over the first row
    arm_copy_q7(row_start, target, dim_im_out*ch_im_in);

    // move over to next row
    row_start += ch_im_in*dim_im_in;

    for ( ;row_start<row_end;row_start+=dim_im_in*ch_im_in) {
      compare_and_replace_if_larger_q7(target, row_start, dim_im_out*ch_im_in);
    }
  }

}

/*
 * bufferA size should be 2*dim_im_out*ch_im_in
 *
 */

void avepool_opt_q7_HWC (
        const q7_t * Im_in,         // input image
        const uint16_t dim_im_in,   // input image dimension
        const uint16_t ch_im_in,    // number of input image channels
        const uint16_t dim_kernel,  // window kernel size
        const uint16_t padding,     // padding sizes
        const uint16_t stride,      // stride
        const uint16_t dim_im_out,  // output image dimension
        q7_t * bufferA,              // a buffer to store temp accumulator
        q7_t * Im_out
) {
  q15_t* buffer = (q15_t*) bufferA;
  int16_t i_ch_in, i_x, i_y;
  int16_t k_x, k_y;
  int16_t count = 0;

  // first does the pooling along x axis
  for(i_y=0;i_y<dim_im_in;i_y++) {

    for(i_x=0;i_x<dim_im_out;i_x++) {
    // for each output pixel
      q7_t* target = Im_in + (i_y*dim_im_in+i_x)*ch_im_in;
      q7_t* win_start;
      q7_t* win_stop;
      if (i_x*stride-padding < 0) {
        win_start = target;
      } else {
        win_start = Im_in + (i_y*dim_im_in + i_x*stride-padding )*ch_im_in;
      }

      if (i_x*stride-padding+dim_kernel >= dim_im_in) {
        win_stop = Im_in + (i_y*dim_im_in + dim_im_in)*ch_im_in;
      } else {
        win_stop = Im_in + (i_y*dim_im_in + i_x*stride-padding+dim_kernel )*ch_im_in;
      }

      // first step is to copy over initial data
      arm_q7_to_q15_no_shift(win_start, buffer, ch_im_in);
      count = 1;

      // start the max operation from the second part
      win_start += ch_im_in;
      for ( ;win_start<win_stop;win_start+=ch_im_in) {
        accumulate_q7_to_q15(buffer, win_start, ch_im_in);
        count ++;
      }
      buffer_scale_back_q15_to_q7(buffer, target, ch_im_in, count);
    }
  }

  // then does the pooling along y axis
  for(i_y=0;i_y<dim_im_out;i_y++) {

  // for each output row      
    q7_t* target = Im_out + i_y*dim_im_out*ch_im_in;
    q7_t* row_start;
    q7_t* row_end;
    // setting the starting row
    if (i_y*stride-padding<0) {
      row_start = Im_in;
    } else {
      row_start = Im_in + (i_y*stride-padding)*dim_im_in*ch_im_in;
    }
    // setting the stopping row
    if (i_y*stride-padding+dim_kernel >= dim_im_in) {
      row_end = Im_in + dim_im_in*dim_im_in*ch_im_in;
    } else {
      row_end = Im_in + (i_y*stride-padding+dim_kernel)*dim_im_in*ch_im_in;
    }

    // copy over the first row
    arm_q7_to_q15_no_shift(row_start, buffer, dim_im_out*ch_im_in);
    count = 1;

    // move over to next row
    row_start += ch_im_in*dim_im_in;

    for ( ;row_start<row_end;row_start+=dim_im_in*ch_im_in) {
      accumulate_q7_to_q15(buffer, row_start, dim_im_out*ch_im_in);
      count ++;
    }
    buffer_scale_back_q15_to_q7(buffer, target, dim_im_out*ch_im_in, count);
  }

}

