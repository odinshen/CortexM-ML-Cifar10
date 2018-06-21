/* --------------------------------------------
 *
 *  Description: softmax and 2-based softmax function
 *
 *  Here, instead of typical e based softmax, we use
 *  2-based softmax, i.e.,:
 *           2^(x_i)
 *  y_i = ------------
 *         sum(e^x_j)
 *
 *  The relative output will be different here.
 *  But mathamatically, the gradient will be the same
 *  with a log(2) scaling factor.
 *
 * -------------------------------------------- */

#include "NNFunctions.h"

void softmax_q7(
        const q7_t* vec_in,        // input vector
        const uint16_t dim_vec,    // dimension of the vector
        q7_t* p_out          // output vector
) {
  q31_t sum;
  int16_t i;
  q15_t min, max;
  max = -257; min = 257;
  for (i=0;i<dim_vec;i++) {
    if (vec_in[i] > max) {
      max = vec_in[i];
    }
    if (vec_in[i] < min) {
      min = vec_in[i];
    }
  }


  // we ignore really small values  
  // anyway, they will be 0 after shrinking
  // to q7_t
  if (max - min > 8) {
    min = max - 8;
  }

  sum = 0;


  for (i=0;i<dim_vec;i++) {
    sum += 0x1<< (vec_in[i] - min);
  }

  for (i=0;i<dim_vec;i++) {
    // we leave 7-bit dynamic range, so that 128 -> 100% confidence
    p_out[i] = (q7_t) __SSAT(((0x1<<(vec_in[i]-min+20))/sum) >> 13 ,8);
  }

}

void softmax_q15(
        const q15_t* vec_in,        // input vector
        const uint16_t dim_vec,    // dimension of the vector
        q15_t* p_out          // output vector
) {
  q31_t sum;
  int16_t i;
  q31_t min, max;
  max = -1*0x100000; min = 0x100000;
  for (i=0;i<dim_vec;i++) {
    if (vec_in[i] > max) {
      max = vec_in[i];
    }
    if (vec_in[i] < min) {
      min = vec_in[i];
    }
  }


  // we ignore really small values  
  // anyway, they will be 0 after shrinking
  // to q7_t
  if (max - min > 16) {
    min = max - 16;
  }

  sum = 0;


  for (i=0;i<dim_vec;i++) {
    sum += 0x1<< (vec_in[i] - min);
  }

  for (i=0;i<dim_vec;i++) {
    // we leave 7-bit dynamic range, so that 128 -> 100% confidence
    p_out[i] = (q15_t) __SSAT(((0x1<<(vec_in[i]-min+14))/sum) ,16);
  }

}

