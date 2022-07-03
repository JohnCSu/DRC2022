from sys import byteorder
import serial
import time
def sendData(data):
    # arduino = serial.Serial(port = 'COM5', baudrate = 9600, timeout = .1)
    # arduino.write(bytes(data , 'utf-8'))
    arduino.write(data.to_bytes(1,byteorder ='big'))
    time.sleep(0.5)
    print(arduino.readline(),end= '\n')

if __name__ == '__main__':

    ask = False
    arduino = serial.Serial(port = 'COM5', baudrate = 9600, timeout = .1)
    while(1):
        if arduino.readline().decode('ascii') == 'on' and ask == False:
            ask = True
            break
    while(1):
        if ask:
            data = input('Enter The data\t')
            print(type(data))
            sendData(int(data))
            