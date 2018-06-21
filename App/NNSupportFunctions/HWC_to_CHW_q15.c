/* ----------------------------------------------------------------------
* Description:   Q15 version of HWC to CHW conversion
*
*
* -------------------------------------------------------------------- */

#include "arm_math.h"
#include "NNSupportFunctions.h"

arm_status HWC_to_CHW_q15(
         const q15_t * HWC_in,          // input image in HWC format
         const uint16_t dim_im_in,      // input image dimention
         const uint16_t ch_im_in,       // number of input image channels
         q15_t * CHW_out               // output imaage in CHW format
) {
  arm_status status;                 /* status of the option */
  int16_t i_x, i_y, i_c;

  for (i_y=0;i_y<dim_im_in;i_y++) {
    for(i_x=0;i_x<dim_im_in;i_x++) {
      for(i_c=0;i_c<ch_im_in;i_c++) {
        CHW_out[i_c*dim_im_in*dim_im_in+i_y*dim_im_in+i_x] 
             = HWC_in[(i_y*dim_im_in+i_x)*ch_im_in+i_c];
      }
    }
  }

  /* set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;

  return (status);
}
