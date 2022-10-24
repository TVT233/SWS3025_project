from __future__ import print_function
from time import sleep
import RPi.GPIO as GPIO
import smbus

I2C_WRITE_ADDR = 0xAE
I2C_READ_ADDR = 0xAF

REG_INTR_STATUS_1 = 0x00
REG_INTR_STATUS_2 = 0x01

REG_INTR_ENABLE_1 = 0x02
REG_INTR_ENABLE_2 = 0x03

REG_FIFO_WR_PTR = 0x04
REG_OVF_COUNTER = 0x05
REG_FIFO_RD_PTR = 0x06
REG_FIFO_DATA = 0x07
REG_FIFO_CONFIG = 0x08

REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG =0x0A
REG_LED1_PA = 0x0C

REG_LED2_PA = 0x0D
REG_PILOT_PA = 0x10
REG_MULTI_LED_CTRL1 = 0x11
REG_MULTI_LED_CTRL2 = 0x12

REG_TEMP_INTR = 0x1F
REG_TEMP_FRAC = 0x20
REG_TEMP_CONFIG = 0x21
REG_PROX_INT_THRESH = 0x30
REG_REV_ID = 0xFE
REG_PART_ID = 0x0F


MAX_BRIGHTNESS = 255

class MAX30102():


    def __init__(self, channel=1, address=0x57, gpio_pin=7):
        print("Channel: {0}, address: {1}".format(channel, address))
        self.address = address
        self.channel = channel
        self.bus = smbus.SMBus(self.channel)
        self.interrupt = gpio_pin


        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.interrupt, GPIO.IN)

        self.reset()

        sleep(1)

        reg_data = self.bus.read_i2c_block_data(self.addressm, REG_INTR_STATUS_1, 1)

        self.setup()


    def shutdowm(self):
        """
        SHUTDOWN THE DEVICE.
        """
        self_bus_write_i2c_block_data(self.address, REG_MODE_CONFIG, [0x80])

    def reset(self):
        """
        reset the device && clear all settings
        do remember to  run setup() agian after doing this
        """
        self.bus.write_i2c_block_data(self.address, REG_MODE_CONFIG, [0x40])

    def setup(self, led_mode=0x03):




        self.bus.write_i2c_block_data(self.address, REG_INTR_ENABLE_1, [0xC0])
        self.bus.write_i2c_block_data(self.address, REG_INTR_ENABLE_2, [0x00])


        self.bus.write_i2c_block_data(self.address, REG_FIFO_WR_PTR, [0x00])

        self.bus.write_i2c_block_data(self.address, REG_OVF_COUNTER, [0x00])

        self.bus.write_i2c_block_data(self.address, REG_FIFO_RD_PTR, [0x00])



        self.bus.write_i2c_block_data(self.address, REG_FIFO_CONFIG, [0x4F])


        self.bus.write_i2c_block_data(self.address, REG_MODE_CONFIG, [led_mode])


        self.bus.write_i2c_block_data(self.address, REG_SPO2_CONFIG, [0x27])


        self.bus.write_i2c_block_data(self.address, REG_LED1_PA, [0X24])

        self.bus.write_i2c_block_data(self.address, REG_LED2_PA, [0x24])

        self.bus.write_i2c_block_data(self.address, REG_PILOT_PA, [0x7f])



    def set_config(self, reg, value):
        self.bus.write_i2c_block_data(self.address, reg, value)
    
    def read_fifo(self):
        """
        This function will read the data register.
        """
        red_led = None
        ir_led = None


        reg_INTR1 = self.bus.read_i2c_block_data(self.address, REG_INTR_STATUS_1, 1)
        reg_INTR2 = self.bus.read_i2c_block_data(self.address, REG_INTR_STATUS_2, 1)


        d = self.bus.read_i2c_block_data(self.address, REG_FIFO_DATA, 6)


        red_led = (d[0] << 16 | d[1] << 8 | d[2]) & 0x03FFFF
        ir_led = (d[3] << 16 | d[4] << 8 | d[5]) & 0x03FFFF

        return red_led, ir_led
    
    def read_sequential(self, amount = 100):
        """
        This fuction will read the red-led and ir-led 'amount' times
        This works as blocking function.
        """
        red_buf = []
        ir_buf = []
        for i in range(amount):
            while(GPIO.input(self.interrupt) == 1):
                pass
            red, ir = self.read_fifo()
            red_buf.append(red)
            ir_buf.append(ir)
        return red_buf, ir_buf

if __name__ == '__main__':
    red, ir = MAX30102().read_sequential(1000)
    print(red)
    print(ir)