# # 모듈= 필요한 것들끼리 부품처럼 잘 만들어진 파일-->함수정의나 클래스들을 포함 하고 있다.
# # 모듈은 내가 모듈을 쓰려는 파일과 같은 경로에 있거나 파이썬 라이브러리들이 모여있는 폴더에 있어야 사용가능

# import theater_module
# theater_module.price(3)#3명이서 영화 보러 갔을 때 가격
# theater_module.price_morning(4) #4명이서 영화 보러 갔을 때 가격
# theater_module.price_soldier(5) #5명이서 영화 보러 갔을 때 가격

# import theater_module as mv # 별명을 붙여서  모듈이름을 줄이는 것 mv를 사용하여 theater_module을 사용 가능
# mv.price(3)
# mv.price_morning(4)
# mv.price_soldier(5)


# from theater_module import *
# # from random import *과 같음

# price(3)
# price_morning(4)
# price_soldier(5)



# from theater_module import price,price_morning #특정 함수만 import 하는 것

# price(5)
# price_morning(4)


# from theater_module import price_soldier as price #price_soldier에 별명 붙인 것

# price(5)


# 패키지->모듈들을 모아놓은 집합

# import travel.thailand # import를 할때는 모듈이나 패키지만 가능함, 클래스나 함수는 불가능  
# trip_to=travel.thailand.ThailandPackage()
# trip_to.detail()

# from travel.thailand import ThailandPackage # from import에서는 가능함 클래스,함수를 
# trip_to=ThailandPackage()
# trip_to.detail() 

# from travel import vietnam
# trip_to=vietnam.VietnamPackage()
# trip_to.detail()


# __all__

# from travel import *
# trip_to=vietnam.VietnamPackage()
# trip_to=thailand.ThailandPackage()
# trip_to.detail()

# *을 쓴다는 것은 travel이란 패키지에 있는 모든 것을 가져온다는 것, 실제로는 개발자가  패키지 안에 포함되어 있는것에서 공개범위 설정해야함

# 모듈 직접 실행


#패키지,모듈 위치

# import inspect
# import random
# print(inspect.getfile(random))
# from travel import *
# print(inspect.getfile(thailand))


# pip install


# 내장 함수-> 따로 import 할 필요없이 바로 사용가능한것

# input= 사용자 입력을 받는 함수

# language=input('무슨 언어를 좋아하세요')
# print('{}은 아주 좋은 언어입니다.'.format(language))

# dir = 어떤 객체를 넘겨줬을 때 그 객체가 어떤 변수와 함수를 가지고 있는지 표시

# print(dir())
# import random #외장 함수
# print(dir())
# import pickle
# print(dir())

# import random
# print(dir(random))


# lst=[1,2,3]
# print(dir(list))

# name='Jim'
# print(dir(name))









