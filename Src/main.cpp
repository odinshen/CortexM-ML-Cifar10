// Includes
#include "main.h"

#include <stm32f7xx_hal.h>
#include <stm32f7xx_hal_cortex.h>
#include <stm32746g_discovery.h>
#include <stm32746g_discovery_camera.h>

#include <D2DResize.h>
#include <cnn.h>
#include <GUI.h>

//Camera
#define CAMERA_RESOLUTION RESOLUTION_R320x240
#define CAMERA_RESOLUTION_WIDTH 320
#define CAMERA_RESOLUTION_HEIGHT 240
uint8_t gCameraBuffer[CAMERA_RESOLUTION_WIDTH*CAMERA_RESOLUTION_HEIGHT*2]; // 565
uint32_t gCameraFramesCounter = 0;

//CNN
__attribute__((section(".sdram_data"))) uint8_t gCNNImageBuffer[CNN_IMAGE_BUFFER_SIZE];

q7_t gScratchBuffer[CNN_SCRATCH_BUFFER_SIZE];
q7_t gColBuffer[CNN_COL_BUFFER_SIZE];
CNN cnn(gScratchBuffer, gColBuffer);

//Resize
__attribute__((section(".sdram_data"))) uint8_t CameraResizeBuffer[CAMERA_RESOLUTION_WIDTH*CAMERA_RESOLUTION_HEIGHT*2];
__attribute__((section(".sdram_data"))) uint8_t CameraResizeWorkBuffer[CAMERA_RESOLUTION_WIDTH*CAMERA_RESOLUTION_HEIGHT*4];

RESIZE_InitTypedef Resize_camera =
{ .SourceBaseAddress = CameraResizeBuffer, // source bitmap Base Address
    .SourcePitch = CAMERA_RESOLUTION_WIDTH, // source pixel pitch
    .SourceColorMode = DMA2D_RGB565, // source color mode
    .SourceX = 0, // souce X
    .SourceY = 0, // sourceY
    .SourceWidth = CAMERA_RESOLUTION_WIDTH - (CAMERA_RESOLUTION_WIDTH - CAMERA_RESOLUTION_HEIGHT), // source width, taking the image from the left corner
    .SourceHeight = CAMERA_RESOLUTION_HEIGHT, // source height
    .OutputBaseAddress = gCNNImageBuffer, // output bitmap Base Address
    .OutputPitch = CNN_RESOLUTION_WIDTH, // output pixel pitch
    .OutputColorMode = DMA2D_RGB888, // output color mode
    .OutputX = 0, // output X
    .OutputY = 0, // output Y
    .OutputWidth = CNN_RESOLUTION_WIDTH, // output width
    .OutputHeight = CNN_RESOLUTION_HEIGHT, // output height
    .WorkBuffer = CameraResizeWorkBuffer // storage buffer
    };

//GUI
__attribute__((section(".sdram_data"))) uint8_t gGuiBuffer1[480*272*4];
__attribute__((section(".sdram_data"))) uint8_t gGuiBuffer2[480*272*4];
const GUI::LayoutConfig_t gGuiLayoutConfig = {
  .CameraFeedTopLeft = {16,16},
  .CameraResolutionWidth = CAMERA_RESOLUTION_WIDTH,
  .CameraResolutionHeight = CAMERA_RESOLUTION_HEIGHT,
  .CNNFeedTopLeft = {256+96,130-32},
  .PieChartCenter = {256+112,130},
  .PieChartRadius = 62,
  .ArmLogoTopLeft = {256+52,16},
  .ImageLabel = {256+70,200},
  .CnnRunTime = {284,228},
  .CnnFramesPerSecond = {375,228}
};
GUI gui(gGuiBuffer1,gGuiBuffer2,gCameraBuffer,gCNNImageBuffer,&gGuiLayoutConfig);


void SystemClock_Config(void);
static void CPU_CACHE_Enable(void);

int main(void)
{
  // Enable the CPU cache
  CPU_CACHE_Enable();

  // Reset of all peripherals and initialise the flash interface and the Systick
  HAL_Init();


  // Configure System Clock
  SystemClock_Config();

  // Initialise GUI
  gui.init();

  // Initialise DMA2D Interrupt
  HAL_NVIC_SetPriority(DMA2D_IRQn, 0x0F, 0);
  HAL_NVIC_EnableIRQ(DMA2D_IRQn);

  // Initialise Camera
  BSP_CAMERA_Init(CAMERA_RESOLUTION);
  BSP_CAMERA_ContinuousStart(gCameraBuffer);

  while(1)
  {
	  asm("nop");
  }
}

void D2D_Resize_Callback(D2D_Stage_Typedef D2D_Stage){

	if(D2D_Stage == D2D_STAGE_ERROR) {

	} else {
		CNN::CNNOutput_t cnn_output = cnn.classify(gCNNImageBuffer);
		gui.drawResults(cnn_output);
	}
}

void BSP_CAMERA_FrameEventCallback(void)
{
	if(gCameraFramesCounter == 50) {
		gCameraFramesCounter = 0;
		BSP_CAMERA_Suspend();
		//TODO: Better with DMA ????
		memcpy(CameraResizeBuffer,gCameraBuffer,CAMERA_RESOLUTION_WIDTH*CAMERA_RESOLUTION_HEIGHT*2);
		BSP_CAMERA_Resume();
		D2D_Resize_Setup(&Resize_camera);
	}
	gCameraFramesCounter++;
}

void BSP_CAMERA_ErrorCallback(DCMI_HandleTypeDef *hdcmi){
  _Error_Handler(__FILE__,__LINE__);
}

//* System Clock Configuration
void SystemClock_Config(void)
{

  RCC_OscInitTypeDef RCC_OscInitStruct;
  RCC_ClkInitTypeDef RCC_ClkInitStruct;

    /**Configure the main internal regulator output voltage
    */
  __HAL_RCC_PWR_CLK_ENABLE();

  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    /**Initializes the CPU, AHB and APB busses clocks
    */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = 16;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 216;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    _Error_Handler(__FILE__, __LINE__);
  }

    /**Activate the Over-Drive mode
    */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    _Error_Handler(__FILE__, __LINE__);
  }

    /**Initializes the CPU, AHB and APB busses clocks
    */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_6) != HAL_OK)
  {
    _Error_Handler(__FILE__, __LINE__);
  }

    /**Configure the Systick interrupt time
    */
  HAL_SYSTICK_Config(HAL_RCC_GetHCLKFreq()/1000);

    /**Configure the Systick
    */
  HAL_SYSTICK_CLKSourceConfig(SYSTICK_CLKSOURCE_HCLK);

  // SysTick_IRQn interrupt configuration
  HAL_NVIC_SetPriority(SysTick_IRQn, 0, 0);
}

// USER CODE END 4

/**
  * @brief  This function is executed in case of error occurrence.
  * @param  None
  * @retval None
  */
void _Error_Handler(const char * file, int line)
{
  // USER CODE BEGIN Error_Handler_Debug
  // User can add his own implementation to report the HAL error return state
  NVIC_SystemReset();
  // USER CODE END Error_Handler_Debug
}

/**
  * @brief  CPU L1-Cache enable.
  * @param  None
  * @retval None
  */
static void CPU_CACHE_Enable(void)
{
  // Enable I-Cache
  SCB_EnableICache();

  // Enable D-Cache
  SCB_EnableDCache();
}

#ifdef USE_FULL_ASSERT

/**
   * @brief Reports the name of the source file and the source line number
   * where the assert_param error has occurred.
   * @param file: pointer to the source file name
   * @param line: assert_param error line source number
   * @retval None
   */
void assert_failed(uint8_t* file, uint32_t line)
{
  // USER CODE BEGIN 6
  /* User can add his own implementation to report the file name and line number,
    ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  // USER CODE END 6

}

#endif

/**
  * @}
  */

/**
  * @}
*/

//
