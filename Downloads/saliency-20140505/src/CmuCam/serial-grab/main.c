#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <png.h>

#include <cc3.h>

static void capture_2serial(void);

int main(void) {

  cc3_uart_init (0,
                 CC3_UART_RATE_115200,
                 CC3_UART_MODE_8N1,
                 CC3_UART_BINMODE_BINARY);

  cc3_camera_init ();

  // use MMC
  cc3_filesystem_init();

  cc3_camera_set_resolution(CC3_CAMERA_RESOLUTION_LOW);
  cc3_timer_wait_ms(1000);

  // init
  while(true) {

    while(!cc3_button_get_state());

    cc3_led_set_state(0, false);

    capture_2serial();

  }

  return 0;
}


void capture_2serial(void)
{
  uint32_t x, y;
  uint32_t size_x, size_y;

  uint32_t time, time2;
  int write_time;

  cc3_pixbuf_load();

  uint8_t *row = cc3_malloc_rows(1);
  uint8_t num_channels = cc3_g_pixbuf_frame.coi == CC3_CHANNEL_ALL ? 3 : 1;

  size_x = cc3_g_pixbuf_frame.width;
  size_y = cc3_g_pixbuf_frame.height;

  time = cc3_timer_get_current_ms();

  putchar(1);
  putchar(size_x>>2);
  putchar(size_y>>2);
  putchar(num_channels);

  for (y = 0; y < size_y; y++) {

    if (y % 4 == 0)
      cc3_led_set_state (0, true);
    else
      cc3_led_set_state (0, false);

    cc3_pixbuf_read_rows(row, 1);

    //if(sw_color_space==HSV_COLOR && num_channels==CC3_CHANNEL_ALL )
    //  cc3_rgb2hsv_row(row,size_x);

    for (x = 0; x < size_x * num_channels; x++) {
      uint8_t p = row[x];

      // avoid confusion from FIFO corruptions
      //if (p < 16) {
      //  p = 16;
      //}
      //else if (p > 240) {
      //  p = 240;
      //}
      putchar (p);
    }


  }

  time2 = cc3_timer_get_current_ms();
  write_time = time2 - time;

  free(row);

  fprintf(stderr, "write_time  %10d\n",
      write_time);
  cc3_led_set_state (0, false);

}
