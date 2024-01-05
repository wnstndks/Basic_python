# # if

# # weather=input('오늘 날씨는 어때요?')
# # if weather=='비'or '눈':
# #     print('우산을 챙기세요')
# # elif weather=='미세먼지':
# #     print('마스크를 챙기세요')
# # else:
# #     print('그냥 오세요')


# # for

# # for i in [0,1,2,3,4]:
#     # print('당신은 %d번째 대기손님입니다'%i)

# #while

# # customer='토르'
# # person='unknown'

# # while person != customer:
# #     print('{}, 커피가 준비 되었습니다.'.format(customer))
# #     person=input('이름이 어떻게 되세요>?')

# #continue break
# absent=[2,5]
# no_book=[7]
# for student in range(1,11):
#     if student in absent:
#         continue
#     elif student in no_book:
#         break
#     print('{} 책 읽어'.format(student))


# students=[1,2,3,4,5]
# print(students)
# students=[i+100 for i in students] 
# print(students)

# 값이 순차적으로 커질때는 range 활용

# continue는 다음 반복을 진행하는 것이고 break는 반복문을 끝내는 것을 의미한다.

# 한줄 for

# students=[1,2,3,4,5]

# students=[i+100 for i in students]
# print(students)
# students

# students=['iron man','torh','Grooto']
# students=[len(i) for i in students]
# students

# students=[i.upper() for i in students]

# from random import *

# cnt=0
# for i in range(1,51):
#     time=randrange(5,51)
#     if 5<=time<=15:
#         print('[O] %d번째 손님(소요 시간: %d)'%(i,time) )
#         cnt+=1
#     else:
#         print('[X] %d번째 손님(소요 시간: %d)'%(i,time) )
    
# 함수란 어떤 박스라고 생각하면 된다.

# def 함수(전달하려는 값)

# def open_account():
#     print('새로운 계좌가 생성되었습니다')
# open_account()

# def deposit(balance,money):
#     print('입금이 완료 잔액은{}원이다'.format(balance+money))
# def withdraw(balance,money):
#     if balance>money:
#         print('출금 완료 잔액은{}'.format(balance-money))
#         return balance-money
#     else:
#         print('출금이 완료X 잔액은{}'.format(balance))
#         return balance

# deposit(1000,500)
# withdraw(1000,5000)





























