# 함수에서 기본값을 설정
# 줄바꿈을 할때는 역슬래쉬를 적고 나서 엔터 쳐주면 하나의 문장으로 처리한다., 터미널을 오른족으로 옮기려면 터미널을 누르고 오른쪽 버튼 클릭 후 오른쪽으로 이동시키면 된다.
# def profile(name,age,main_lang):
#         print('이름:{}\t나이:{}\t주 사용 언어:{}'.format(name,age,main_lang))

# profile('유재석',20,'파이썬')
# profile('김태호',25,'자바')

# 같은 학교 같은 학년 같은 반 같은 수업.
# def profile(name,age=17,main_lang='python'):
#         print('이름:{}\t나이:{}\t주 사용 언어:{}'\
#                 .format(name,age,main_lang))

# profile('안준수')
# profile('김준수')
# profile('최준수')
# profile('선준수')


# 키워드값을 이용한 함수 호출

# def profile(name,age,main_lang):
#         print('이름:{}\t나이:{}\t주 사용 언어:{}'\
#                  .format(name,age,main_lang))

# profile(name='유재석',main_lang='파이썬',age=20)


# 가변 인자 이용한 함수 호출
# end를 쓸시 줄바꿈 없어짐
# def profile(name,age,lang1,lang2,lang3,lang4,lang5):
#         print('이름:{}\t나이:{}\t'.format(name,age),end='')
#         print(lang1,lang2,lang3,lang4,lang5)

# def profile(name,age,*language):
#         print('이름:{}\t나이:{}\t'.format(name,age),end='')
#         for lang in language:
#                 print(lang, end='')
#         print()
# profile('유재석',20,'파이썬','자바','씨','씨샵','씨쁠쁠')
# profile('김재석',25,'kotlin','swift','','','')



# 지역변수=함수 내에서만 쓸수 있는 것 함수 호출이 끝나면 사라짐 와 전역변수=프로그램 내 모든 곳에서 사용 가능
# tab을 쳐서 자동완성하기

# gun=10

# def checkpoint(soldiers):
#         global gun #전역 공간에 있는 gun 사용
#         gun=gun-soldiers  #gun은 함수 내에서 만들어진 gun 이므로 초기화가 안되었기때문에 사용 불가능
#         print('[함수 내] 남은 총:{}'.format(gun))
# def checkpoint_ret(gun,soldiers):
#         gun=gun-soldiers
#         print('[함수 내] 남은 총:{}'.format(gun))
#         return gun



# print('전체 총:%d'%gun)
# # checkpoint(2)
# gun=checkpoint_ret(gun,2)
# print('남은 총:%d'%gun)






# def std_weight(height,gender):
#         global new_height
#         if gender=='남자':
#                 new_height=height*height/10000*22
#                 print('키{}cm {}이 표준 체중은 {}kg입니다.'.format(height,gender,new_height))
#         else:
#                 new_height=height*height/10000*21
#                 print('키{}cm {}이 표준 체중은 {}kg입니다.'.format(height,gender,new_height))

# height=int(input('키을 입력하세요'))
# gender=(input('성별을 입력하세요'))

# std_weight(height,gender)


# round 함수로 반올림 가능

# def std_weight(height,gender):
#         if gender=='남자':
#                 return height*height*22
#         else:
#                 return height*height*21


# height=175
# gender='남자'

# weight=round(std_weight(height/100,gender),2)
# print('키{}cm {}이 표준 체중은 {}kg입니다.'.format(height,gender,weight))


















