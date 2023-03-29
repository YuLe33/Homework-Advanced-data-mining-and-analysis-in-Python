def my_func(x1,x2,x3):
    
    con=((type(x1)!=float) or (type(x2)!=float) or (type(x3)!=float))
    if con== True:
        pass
        if x1==0 and x2==0 and x3==0:
            return "Not a number – denominator equals zero"
        
        elif (type(x1)== int) or (type(x2) == int) or (type(x3) == int):
            return "Error: parameters should be float"
        
        else:
            return None  
    
    else:
        a=x1+x2+x3
        if a==0:
            return "Not a number – denominator equals zero" 
        b=x2+x3
        return float((a*b*x3)/a)

print(my_func(31.1,4.2, 2.2))
print(my_func(8.0,0.0,-8.0))
print(my_func(31.1,10, 2.2))
print(my_func(31.1,list("john"), 2.2))