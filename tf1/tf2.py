# 변수 선언 후 사용하기
# tf.Variable()

import tensorflow as tf

print(tf.constant(1.0)) # 고정된 데이터값(ex.상수)을 기억
# vs
f=tf.Variable(1.0) #변수형 0-d 텐서 : scala
v= tf.Variable(tf.ones(2,)) # 1-d 텐서 : vector
m= tf.Variable(tf.ones(2,1)) # 2-d 텐서 : matrix
print(f)
print(v)
print(m)
print()

#치환
v1=tf.Variable(1)
v1.assign(10)
print('v1 : ', v1,v1.numpy())
print()

w=tf.Variable(tf.ones(shape=(1,)))
b=tf.Variable(tf.ones(shape=(1,)))
w.assign([2])
b.assign([3])

def func1(x): #얘는 텐서를 가지고 있지칸 파이썬을 쓰기에 연산속도 느림
    return w*x+b

print(func1(3))

@tf.function #auto graph 기능 : 별도의 텐서 영역에서 연산을 수행. tf.Graph+tf.Session -> 영역내에서 텐서를 수행할떄 그래프를 만들고 그 그래프를 세션이 수행함으로서 텐서를 수행하는 것
def func2(x):
    return w*x+b

print(func2(3))
#텐서는 클래스나 함수를 만들어서 수행시키는 게 좋다 그러면 세션에 의해 수행되기에 빨라진다.

print('Variable의 치환 / 누적')
aa=tf.ones(2,1)
print(aa.numpy())
m=tf.Variable(tf.zeros(2,1))
print(m.numpy()) 
m.assign(aa) #치환 1값으로 치환한것
print(m.numpy()) 

#값 누적시키기
m.assign_add(aa)
print(m.numpy())

m.assign_sub(aa) 
print(m.numpy())
print()

#구조적인것
print('---------------'*10)
g1=tf.Graph() #기본적으로 내장된 그래프 

with g1.as_default():
    c1=tf.constant(1,name="c_one") #c1은 name이 c_one이란 이름의 정수 1을 가진 상수(고정된 값을 기억)다.
    print(c1) #c1은 텐서값을 그대로 유지하고 있음 , c1은 실행의 대상을 가리키고 있는 것일 뿐, 
    print(type(c1))
    print(c1.op)
    print('--------')
    print(g1.as_graph_def())


print('---------------'*10)
g2=tf.Graph() #기본적으로 내장된 그래프 
  
with g2.as_default():
    v1=tf.Variable(initial_value=1,name='v1')
    print(v1)
    print(type(v1))
    print(v1.op)

print(g2.as_graph_def())
# 포인터를 가지고 있는 것 variable 하나 선언했지만 여러 정보를 가지고 있다.- 그래프 영역에서 연산되고 있음, 그래프 객체를 꺼냈음
# 그래프 객체에서 내부적으로 벌어지고 있다는 것을 보여준다.











