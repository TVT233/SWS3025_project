import max30102
m = max30102.MAX30102()
red, ir = m.read_sequential(1000)
print(red)
print(ir)