#include "D2DResize.h"

#include <stm32f746xx.h>
#include <stm32f7xx_hal_dma2d.h>

static DMA2D_HandleTypeDef hDma2dHandler;

// Setup FG/BG address and FGalpha for linear blend, start DMA2D
void D2D_Blend_Line(DMA2D_HandleTypeDef *);
void D2D_TranferComplete_handle(DMA2D_HandleTypeDef *);
void D2D_TranferError_handle(DMA2D_HandleTypeDef *);

// resize loop parameter
typedef struct {
	uint32_t Counter;           // Loop Counter
	uint32_t BaseAddress;       // Loop Base Address
	uint32_t BlendIndex;        // Loop Blend index (Q21)
	uint32_t BlendCoeff;        // Loop Blend coefficient (Q21)
	uint16_t SourcePitchBytes;  // Loop Source pitch bytes
	uint16_t OutputPitchBytes;  // Loop Output pitch bytes
	uint32_t OutputWidth;		  // Loop Output width
	uint32_t OutputHeight;	  // Loop OutputHeight
	uint32_t OutputAddress;     // Loop OutputAddress
} D2D_Loop_Typedef;

//Current resize stage
D2D_Stage_Typedef D2D_Loop_Stage = D2D_STAGE_IDLE;

//First and second loop parameter
D2D_Loop_Typedef D2D_1st_Loop, D2D_2nd_Loop;

//current parameter pointer
D2D_Loop_Typedef* D2D_Loop = &D2D_1st_Loop;

// storage of misc. parameter for 2nd loop DMA2D register setup
struct {
	uint32_t OutputBaseAddress; // output bitmap Base Address
	uint16_t OutputColorMode;   // output color mode
	uint16_t OutputPitch;       // output pixel pitch
	uint16_t SourceWidth;       // source width
} D2D_Misc_Param;

/*

 D2D_Resize_Setup() Setup and start the resize process.
 parameter:         RESIZE_InitTypedef structure
 return value:      D2D_STAGE_SETUP_DONE if process start or D2D_STAGE_SETUP_BUSY
 when a resize process already in progress

 */
D2D_Stage_Typedef D2D_Resize_Setup(RESIZE_InitTypedef* R) {
	uint16_t PixelBytes, PitchBytes;
	uint32_t BaseAddress;

	const uint16_t BitsPerPixel[6] = { 32, 24, 16, 16, 16, 8 };

	// Test for loop already in progress
	if (D2D_Loop_Stage != D2D_STAGE_IDLE)
		return (D2D_STAGE_SETUP_BUSY);

	// 1st loop parameter init
	PixelBytes = BitsPerPixel[R->SourceColorMode] >> 3; //get the number of byte per pixel
	PitchBytes = R->SourcePitch * PixelBytes; //the total number of byte for a row
	BaseAddress = (uint32_t) R->SourceBaseAddress + R->SourceY * PitchBytes + R->SourceX * PixelBytes; //set be base address to the correx Y,X (Y: height, X:Widtch)

	D2D_1st_Loop.Counter = R->OutputHeight;
	D2D_1st_Loop.SourcePitchBytes = PitchBytes; // total number of byte per row
	D2D_1st_Loop.OutputPitchBytes = R->SourceWidth << 2; //output pitch = SourceWidht * 4; 4 is the number of byte for the pixel in the intermediate buffer
	D2D_1st_Loop.BaseAddress = BaseAddress; // set image start location
	D2D_1st_Loop.BlendCoeff = ((R->SourceHeight - 1) << 21) / R->OutputHeight; //SourceHeight / OutputHeight in Q21 fixed point notation
	D2D_1st_Loop.BlendIndex = D2D_1st_Loop.BlendCoeff >> 1; //blend index = blendCoeff / 2
	D2D_1st_Loop.OutputAddress = (uint32_t) R->WorkBuffer;
	D2D_1st_Loop.OutputWidth = R->SourceWidth;
	D2D_1st_Loop.OutputHeight = 1;

	// 2nd loop parameter init
	PixelBytes = BitsPerPixel[R->OutputColorMode] >> 3; //check the byte per pixel for the output picture
	PitchBytes = R->OutputPitch * PixelBytes; //the output pitch (the size of columns to change for the output)
	BaseAddress = (uint32_t) R->OutputBaseAddress + R->OutputY * PitchBytes + R->OutputX * PixelBytes; //set the output base address

	D2D_2nd_Loop.Counter = R->OutputWidth; //Set the counter for the second loop to the width of the output
	D2D_2nd_Loop.SourcePitchBytes = 4; //source for the work image pixel width is encoded in ARGB8888 so 4 byte
	D2D_2nd_Loop.OutputPitchBytes = PixelBytes; //the size
	D2D_2nd_Loop.BaseAddress = (uint32_t) R->WorkBuffer; //the source of the 2nd loop is the work buffer
	D2D_2nd_Loop.BlendCoeff = ((R->SourceWidth - 1) << 21) / R->OutputWidth; //SourceWidth -1 / OutputWidth in Q21 notation
	D2D_2nd_Loop.BlendIndex = D2D_2nd_Loop.BlendCoeff >> 1; //blend index = blend coeff / 2
	D2D_2nd_Loop.OutputAddress = BaseAddress;
	D2D_2nd_Loop.OutputWidth = 1;
	D2D_2nd_Loop.OutputHeight = R->OutputHeight;

	// Mist partameter
	D2D_Misc_Param.OutputBaseAddress = BaseAddress; //set the output buffer to the output position
	D2D_Misc_Param.OutputColorMode = R->OutputColorMode; //the output collor mode
	D2D_Misc_Param.OutputPitch = R->OutputPitch;
	D2D_Misc_Param.SourceWidth = R->SourceWidth;
	// start first loop stage
	D2D_Loop = &D2D_1st_Loop; //set the parameter for the first loop
	D2D_Loop_Stage = D2D_STAGE_FIRST_LOOP;

	//set DMA2D instance
	hDma2dHandler.Instance = DMA2D;
	hDma2dHandler.XferCpltCallback = D2D_TranferComplete_handle;
	hDma2dHandler.XferErrorCallback = D2D_TranferError_handle;
	//set basic parameter for first iteration
	hDma2dHandler.Init.Mode = DMA2D_M2M_BLEND;
	//configure the DMA2D for the first loop
	//hdma2d.Init.OutputOffset = (uint32_t)R->WorkBuffer;
	hDma2dHandler.Init.ColorMode = DMA2D_OUTPUT_ARGB8888;
	HAL_DMA2D_Init(&hDma2dHandler);

	//configure the Layers for the first loop
	//background layer
	hDma2dHandler.LayerCfg[0].AlphaMode = DMA2D_REPLACE_ALPHA;
	hDma2dHandler.LayerCfg[0].InputColorMode = R->SourceColorMode;
	hDma2dHandler.LayerCfg[0].InputAlpha = 0xFF;
	hDma2dHandler.LayerCfg[0].InputOffset = 0;
	HAL_DMA2D_ConfigLayer(&hDma2dHandler, 0);

	//forgraound layer
	hDma2dHandler.LayerCfg[1].AlphaMode = DMA2D_REPLACE_ALPHA;
	hDma2dHandler.LayerCfg[1].InputColorMode = R->SourceColorMode;
	hDma2dHandler.LayerCfg[1].InputAlpha = 0x00;
	hDma2dHandler.LayerCfg[1].InputOffset = 0;
	HAL_DMA2D_ConfigLayer(&hDma2dHandler, 1);

	D2D_Blend_Line(&hDma2dHandler);
	return (D2D_STAGE_SETUP_DONE);
}

D2D_Stage_Typedef D2D_Resize_Stage(void) {
	return (D2D_Loop_Stage);
}

// Setup FG/BG address and FGalpha for linear blend, start DMA2D
void D2D_Blend_Line(DMA2D_HandleTypeDef * hdma2d) {
	uint32_t FirstLine, FGalpha;

	// Integer part of BlendIndex (Q21) is the first line number
	FirstLine = D2D_Loop->BlendIndex >> 21;  //integer part of the index
	// calculate and setup address for first and 2nd lines
	uint32_t bg_memory_address = D2D_Loop->BaseAddress + FirstLine * D2D_Loop->SourcePitchBytes;
	uint32_t fg_memory_address = bg_memory_address + D2D_Loop->SourcePitchBytes;

	// 8 MSB of fractional part as FG alpha (Blend factor)
	FGalpha = D2D_Loop->BlendIndex >> 13 & 0xFF; //not this will change the Q notation to Q8 without integer part
	hdma2d->LayerCfg[1].InputAlpha = FGalpha;
	HAL_DMA2D_ConfigLayer(hdma2d, 1);

	// restart DMA2D transfer
	HAL_DMA2D_BlendingStart_IT(hdma2d, fg_memory_address, bg_memory_address, D2D_Loop->OutputAddress,
			D2D_Loop->OutputWidth, D2D_Loop->OutputHeight);
	//DMA2D_StartTransfer();
}

void D2D_TranferComplete_handle(DMA2D_HandleTypeDef * hdma2d) {
	// Test for loop in progress
	if (D2D_Loop_Stage != D2D_STAGE_IDLE) {
		// decrement loop counter and if != 0 process loop row
		if (--D2D_Loop->Counter) {
			// Update output memory address
			D2D_Loop->OutputAddress += D2D_Loop->OutputPitchBytes;
			// Add BlenCoeff to BlendIndex
			D2D_Loop->BlendIndex += D2D_Loop->BlendCoeff;
			// Setup FG/BG address and FGalpha for linear blend, start DMA2D
			D2D_Blend_Line(hdma2d);
		} else {
			// else test for current D2D Loop stage
			if (D2D_Loop_Stage == D2D_STAGE_FIRST_LOOP) {

				hdma2d->Init.ColorMode = D2D_Misc_Param.OutputColorMode;
				hdma2d->Init.OutputOffset = D2D_Misc_Param.OutputPitch - 1;
				HAL_DMA2D_Init(hdma2d);

				hdma2d->LayerCfg[0].AlphaMode = DMA2D_REPLACE_ALPHA;
				hdma2d->LayerCfg[0].InputColorMode = DMA2D_INPUT_ARGB8888;
				hdma2d->LayerCfg[0].InputAlpha = 0xFF;
				hdma2d->LayerCfg[0].InputOffset = D2D_Misc_Param.SourceWidth - 1;
				HAL_DMA2D_ConfigLayer(hdma2d, 0);

				hdma2d->LayerCfg[1].AlphaMode = DMA2D_REPLACE_ALPHA;
				hdma2d->LayerCfg[1].InputColorMode = DMA2D_INPUT_ARGB8888;
				hdma2d->LayerCfg[1].InputAlpha = 0x00;
				hdma2d->LayerCfg[1].InputOffset = D2D_Misc_Param.SourceWidth - 1;
				HAL_DMA2D_ConfigLayer(hdma2d, 1);

				// start 2nd loop stage
				D2D_Loop = &D2D_2nd_Loop;
				D2D_Loop_Stage = D2D_STAGE_2ND_LOOP;
				// Setup FG/BG address and FGalpha for linear blend, start DMA2D
				D2D_Blend_Line(hdma2d);
			} else {
				// else resize complete
				D2D_Resize_Callback(D2D_STAGE_DONE);
				// reset to idle stage
				D2D_Loop_Stage = D2D_STAGE_IDLE;
			}
		}
	}
}

void D2D_TranferError_handle(DMA2D_HandleTypeDef * hdma2d) {
	// Test for resize loop in progress
	if (D2D_Loop_Stage != D2D_STAGE_IDLE) {
		// resize error callback
		D2D_Resize_Callback(D2D_STAGE_ERROR);
		// reset to IDLE stage
		D2D_Loop_Stage = D2D_STAGE_IDLE;
	}
}

void D2D_IRQHandler(void) {
	HAL_DMA2D_IRQHandler(&hDma2dHandler);
}

__weak void D2D_Resize_Callback(D2D_Stage_Typedef D2D_Stage) {
	// Halt on DMA2D Transfer error
	while (D2D_Stage == D2D_STAGE_ERROR)
		;
}

// D2D_resize.c - END of File
