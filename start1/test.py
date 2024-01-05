'''
문제1.
a=27359242
if a%11==0:
    print('a는 11의 배수입니다.')
else:
    print('a는 11의 배수가 아닙니다.')
'''



'''
문제2.
class cat:
    def __init__(self,name,weight):
        self.name=name
        self.weight= weight
    def bark(self):
        print('요롱이 몸무게: {}'.format(self.weight))
        self.weight-=1

x= cat('요롱이',12)
x.bark()
x.bark()'''

'''
문제3.
age=int(input('나이를 입력하세요'))

if age>=8:
    print('탑승 가능')
else:
    print('탑승 불가능')
'''   

'''
문제4.
class airplain:
    def __init__(self,start,end,hour):
        self.start=start
        self.end=end
        self.hour=hour
    def cost(self):
        print('{}부터 {}까지 여행 경비는{}만원 입니다.'\
            .format(self.start,self.end,self.hour*10))

x=airplain('서울','제주',1)
x.cost()
x=airplain('서울','도쿄',2)
x.cost()
x=airplain('서울','베이징',3)
x.cost()
'''

'''
문제5.
score=[30,55,82,91,89,72,57,93,68,80]
result=0

for student_score in score:
    if 90>student_score>=80:
        result+=student_score**2
        print(student_score)

print('결과:',result)
'''

'''
문제6.
sentence='비트코인-이더리움-도지코인'
list1=['에이다','리플']
list2= sentence.split('-')
list2.remove('이더리움')
list2.reverse()
list=list1+list2
print(list)
'''

'''
문제7.
for i in range(10):
    if i%2==0:
        if i%5==0:
            print('%d, 건강하세요'%i)
        else:
            print('%d, 안녕하세요'%i)
    else:
        if i%5==0:
            print('%d, 건강하세요'%i)
        else:
            print('%d, 반갑습니다'%i)
'''
'''
문제8.
import dinner as dn

dn.beef(2)-1
dn.pork(3)
dn.pasta(2)

print('저녁식사 비용은'+str(beef+pork+pasta)+'만원')
'''


'''
문제9.
try:
    a = int(input("a 를 입력하세요: "))
    list_prob9 = [2, 4, 6]
    if a>=3:
        raise ValueError
    list_prob9[a]=10
    print(list_prob9)
except ValueError:
    print('list_prob9가 해당 index를 갖지 않습니다.')
'''

'''
문제10.
total=0

cof_file=open('coffee.txt','r',encoding='utf8')
print(cof_file.read())




'''



