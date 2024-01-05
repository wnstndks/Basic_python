
# 리스트
subway=['유재석','조세호','박명수']
print(subway)

print(subway.index('조세호'))

subway.append('하하')
print(subway)

subway.insert(1,'정형돈')
print(subway)

subway.pop()
print(subway)

subway.pop()
print(subway)
subway.pop()
print(subway)

subway.append('유재석')

print(subway.count('유재석'))

num_list=[5,2,3,4,1]

num_list.sort()
print(num_list)

num_list.reverse()
print(num_list)


mix_list=['me',5,'you']
num_list=[5,3,1,56,7]

num_list.extend(mix_list)
print(num_list)



#사전(dictionary)
# key 와 value의 형태

cabinet={3:'유재석',100:'김태호'}
print(cabinet[3])
print(cabinet[100])
print(cabinet.get(3))
print(cabinet[5])
print(cabinet.get(5))
print('hi')
print(cabinet.get(5,'사용가능'))

print(3 in cabinet)
print(5 in cabinet)

cabinet={'A-3':'유재석','B-100':'김태호'}

print(cabinet['A-3'])
print(cabinet)
cabinet['C-20']='조세호'
print(cabinet)
cabinet['A-3']='김종국'
print(cabinet)

del cabinet['A-3']
print(cabinet)

print(cabinet.keys())

print(cabinet.values())

print(cabinet.items())

cabinet.clear()

print(cabinet)


#튜플= 변경 불가능

menu=('돈까스','치즈까스')
print(menu[0])
print(menu[1])

# menu.add('생선까스')

name,age,hobby='김종국',20,'운동'

print(name,age,hobby)

#집합->중복 안됨,순서 없음

my_set={1,2,3,3,3}
print(my_set)

java={'유재석','김태호','양세형'}
python=set(['유재석','박명수'])

print(java & python)
print(java.intersection(python))

print(java|python)
print(java.union(python))

print(java-python)
print(java.difference(python))

python.add('김태호')
print(python)


java.remove('김태호')
print(java)



# 자료구조의 변경

menu={'커피','우유','주스'}
print(menu,type(menu))

menu=list(menu)
print(menu,type(menu))

menu=tuple(menu)
print(menu,type(menu))

menu=set(menu)
print(menu,type(menu))


#문제

from random import *

users=range(1,21) #1붙 20까지 숫자 생성
print(type(users))
users=list(users)
print(type(users))
print(users)
shuffle(users)
print(users)

winners=sample(users,4) #한명은 치킨,3명은 커피

print('--당첨자발표--')
print('치킨 당첨자:{}'.format(winners[0]))
print('커피 당첨자:{}'.format(winners[1:]))
print('--축하합니다--')
























































































































