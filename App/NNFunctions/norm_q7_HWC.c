/* ----------------------------------------------------------------------    
* Description:   Q7 version of normalization
*    
* -------------------------------------------------------------------- */
#include "NNFunctions.h"

void norm_q7_HWC (
        const q7_t * Im_in,          // input image
        const uint16_t dim_im_in,    // input image dimention
        const uint16_t ch_im_in,     // number of input image channels
        const uint16_t window_size,  //
        const norm_type type,     // type of norm operation
        q7_t * Im_out
) {
  int16_t i_ch_in, i_x, i_y, l, kx;
  int half_win = floor(window_size/2);
  int LRN = (type==INTER_CHANNEL) ? window_size : window_size*window_size;
  float alpha = 5e-5;
  float beta = 0.75;

  for(i_ch_in=0;i_ch_in<ch_im_in;i_ch_in++) {
    for(i_y=0;i_y<dim_im_in;i_y++) {
      for(i_x=0;i_x<dim_im_in;i_x++) {
        int sum_sq = 0;
        for(l=0;l<window_size;l++) {
          if(type == INTRA_CHANNEL) {
            // cross channel normalization
            if ( i_ch_in-half_win+l >= 0 && i_ch_in-half_win+l < ch_im_in) {
              sum_sq += Im_in[(i_ch_in-half_win+l) + (i_y*dim_im_in+i_x)*dim_im_in] * Im_in[(i_ch_in-half_win+l) + (i_y*dim_im_in+i_x)*dim_im_in];
            }
          } else {
            // within channel normalization
            if ( i_y-half_win+l >=0 && i_y-half_win+l < dim_im_in) {
              for (kx=0;kx<window_size;kx++) {
                if (i_x-half_win+kx >= 0 && i_x-half_win+l < dim_im_in) {
                  sum_sq += Im_in[i_ch_in + ((i_y-half_win+l) * dim_im_in + i_x-half_win+l) * dim_im_in] * Im_in[i_ch_in + ((i_y-half_win+l) * dim_im_in + i_x-half_win+l) * dim_im_in];
                }
              }
            } 
          }
        }
        Im_out[i_ch_in+(i_y*dim_im_in+i_x)*dim_im_in] = (q7_t)Im_in[i_ch_in+(i_y*dim_im_in+i_x)*dim_im_in] / pow(1+alpha/LRN*sum_sq, beta);
      }
    }
  }

}
