from uart_protocol import UART

HEADER_1st = 0xAA
HEADER_2st = 0X55

uart = UART(HEADER_1st, HEADER_2st)

data_flow = [0xAA, 0x55, 5, 10, 15,
          0xAA, 0x55, 20, 25, 30]
print(data_flow)

for frame in data_flow:
    result = uart.get_data(frame)
    if result:
        print(result)

frame = uart.send_data(1,2,44)
print(frame)