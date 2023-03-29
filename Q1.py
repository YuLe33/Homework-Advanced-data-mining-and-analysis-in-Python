def my_func(x1,x2,x3):
    
    a=((type(x1)!=float) or (type(x2)!=float) or (type(x3)!=float))
    if a== True:
        pass
        if x1==0 and x2==0 and x3==0:
            return "Not a number – denominator equals zero"
        
        elif (type(x1)== int) or (type(x2) == int) or (type(x3) == int):
            return "Error: parameters should be float"

        else:
            return None  

    else:
        a=x1+x2+x3
        b=x2+x3
        return float((a*b*x3)/a)
    
print(my_func(31.1,list("john"), 2.2))