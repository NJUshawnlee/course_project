
def diedai(x1,x2):
    x3 = x2 - (x2**3-3*x2-1)*(x2-x1)/((x2**3-3*x2-1)-(x1**3-3*x1-1))
    return x3
x2 = diedai(2,1.9)
print(x2)
x3 = diedai(1.9,x2)
print(x3)

def paowuxian(x0,x1,x2):
    fx0 = x0**3-3*x0-1
    fx1 = x1**3-3*x1-1
    fx2 = x2**3-3*x2-1
    d10 = (fx1-fx0)/(x1-x0)
    d20 = (fx2-fx0)/(x2-x0)
    d21 = (fx2-fx1)/(x2-x1)
    d210 = (d20-d10)/(x2-x1)
    w = d21 + d210*(x2 - x1)
    x3 = x2 - (2*fx2)/(w+(w**2-4*fx2*d210)**0.5)
    print(fx0,fx1,fx2,d10,d20,d21,d210,w,x3)
    return x3

x3 = paowuxian(1,3,2)
x4 = paowuxian(3,2,x3)

