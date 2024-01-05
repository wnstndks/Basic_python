#  예외 처리

# try: 
#     print('나누기 전용 계산기입니다')
#     nums=[]
#     nums.append(int(input('첫번째 숫자를 입력하세요: ')))
#     nums.append(int(input('두번째 숫자를 입력하세요: ')))
#     # nums.append(int(nums[0]/nums[1]))
#     print('{} / {} ={}'.format(nums[0],nums[1],nums[2]))
# except ValueError:
#     print('에러, 잘못된 값을 입력하였습니다.')
# except ZeroDivisionError as err:
#     print(err)
# except Exception as err:
#     print('알 수 없는 에러가 발생하였습니다.')
#     print(err)

# 예외 발생시키기


# try:
#     print('한 자리 숫자 나누기 전용 계산기입니다')
#     num1=(int(input('첫번째 숫자를 입력하세요: ')))
#     num2=(int(input('두번째 숫자를 입력하세요: ')))
#     if num1>=10 or num2>=10:
#         raise ValueError 
#     print('{} / {} = {}'.format(num1,num2,int(num1/num2)))
# except ValueError:
#     print('잘못된 값을 입력하였습니다. 한자리 숫자만 입력하세요.')


# 사용자 정의 예외처리

# class BigNumberError(Exception):
#     def __init__(self, msg):
#         self.msg=msg
#     def __str__(self):
#         return self.msg

# try:
#      print('한 자리 숫자 나누기 전용 계산기입니다')
#      num1=(int(input('첫번째 숫자를 입력하세요: ')))
#      num2=(int(input('두번째 숫자를 입력하세요: ')))
#      if num1>=10 or num2>=10:
#          raise BigNumberError('입력 값:{}, {}'.format(num1,num2))
#      print('{} / {} = {}'.format(num1,num2,int(num1/num2)))
# except ValueError:
#      print('잘못된 값을 입력하였습니다. 한자리 숫자만 입력하세요.')
# except BigNumberError as err:
#      print('에러가 발생하였습니다. 한자리 숫자만 입력하세요.')
#      print(err)


#  예외 발생시키기-> 의도적으로 특정조건을 발생시킴으로서 except 구문으로 가게 할 수 있음 

# try:
#     print('한자리 숫자 나누기 전용 계산기입니다.')
#     num1=int(input('첫 번째 숫자를 입력하시오:'))
#     num2=int(input('두 번째 숫자를 입력하시오:'))
#     if num1>=10 or num2>=10:
#         raise ValueError
#     print('{}/{}={}'.format(num1,num2,int(num1/num2)))
# except ValueError:
#     print('잘 못된 값을 입력하였습니다. 한자리 숫자만 입력하세요')


# finally-> 예외처리에서 무조건 실행되는 부분
class BigNumberError(Exception):
     def __init__(self, msg):
         self.msg=msg
     def __str__(self):
        return self.msg
try:
      print('한 자리 숫자 나누기 전용 계산기입니다')
      num1=(int(input('첫번째 숫자를 입력하세요: ')))
      num2=(int(input('두번째 숫자를 입력하세요: ')))
      if num1>=10 or num2>=10:
          raise BigNumberError('입력 값:{}, {}'.format(num1,num2))
      print('{} / {} = {}'.format(num1,num2,int(num1/num2)))
except ValueError:
      print('잘못된 값을 입력하였습니다. 한자리 숫자만 입력하세요.')
except BigNumberError as err:
      print('에러가 발생하였습니다. 한자리 숫자만 입력하세요.')
      print(err)
finally:
    print('계산기를 이용해 주셔서 감사합니다.')
