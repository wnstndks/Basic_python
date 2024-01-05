# # 표준 입출력

# # print('python','java','Javascript',sep='/ ',end='?')
# # print('무엇이 더 재밌을까')

# # import sys
# # print('python','java',file=sys.stdout) # 표준 출력으로 문장 출력
# # print('python','java',file=sys.stderr) # 표준 에러로 찍히는 것

# # scores={'수학':0,'영어':50,'코딩':100}
# # for subject,score in scores.items(): #.items를 쓸때는 키와 value 둘다 나오게 할 수 있다.
# #     print(subject.ljust(8),str(score).rjust(4),sep=':')

# # for num in range(1,21):
# #     print('대기번호:'+str(num).zfill(3))


# # 표준 입력

# # answer=input('값을 입력하세요')
# # print(type(answer))
# # print('입력하신 값은'+answer+'입니다')

# # 사용자 입력을 통해 값을 받게 되면 항상 문자열 형태로 입력 받게 됨 따라서 int로 감싸주기

# #다양한 출력 포맷


# # 빈 자리는 빈공간으로 두고, 오른쪽 정렬을 하되,총 10자리 공간을 확보

# print('{0: >10}'.format(500))

# # 양수일 땐 +로 표시, 음수일 때는 -로 표시4
# print('{0: >+10}'.format(500))
# print('{0: >+10}'.format(-500))

# # 왼쪽 정렬하고, 빈칸으로 _채움
# print('{0:_<10}'.format(500))

# # 3자리마다 콤마를 찍어주기
# print('{0:,}'.format(1000000000000))
# # 3자리마다 콤마를 찍어주기,+- 부호도 붙이기
# print('{0:+,}'.format(1000000000000))
# print('{0:+,}'.format(-1000000000000))

# # 3자리마다 콤마를 찍어주기,+- 부호도 붙이기,자릿수 확보하기, 빈자리는 ^로 채우기
# print('{0:^<+30,}'.format(1000000000000000))
# # 소수점으로 출력
# print('{0:f}'.format(5/3))
# # 특정소수점까지 출력(소수점 3째 짜리에서 반올림)
# print('{0:.2f}'.format(5/3))


# 파일 입출력
# score_file=open('score.txt','w',encoding='utf8')
# print('수학:0',file=score_file)
# print('영어:50',file=score_file)
# score_file.close()

# score_file=open('score.txt','a',encoding='utf8')
# score_file.write('과학:80')
# score_file.write('\n코딩:100')
# score_file.close()

# score_file=open('score.txt','r',encoding='utf8')
# print(score_file.read())
# score_file.close()

# score_file=open('score.txt','r',encoding='utf8')
# print(score_file.readline(),end='') #줄별로 읽기, 한줄읽고 커서는 다음줄로 이동
# print(score_file.readline(),end='')
# print(score_file.readline(),end='')
# print(score_file.readline(),end='')
# score_file.close()

# score_file=open('score.txt','r',encoding='utf8')
# while True:
#     line=score_file.readline()
#     if not line:
#         break
#     print(line,end='')
# score_file.close()


# score_file=open('score.txt','r',encoding='utf8')
# lines=score_file.readlines() #리스트 형태로 저장
# for line in lines:
#     print(line,end='')
# score_file.close()


# pickle->프로그램 상에서 우리가 사용하고 있는 데이터를 파일형태로 저장하는 것

import pickle
# profile_file=open('profile.pickle','wb')
# profile={'name':'conan','age':30,'hobby':['soccer','golf','swimming']}
# print(profile)
# pickle.dump(profile,profile_file)# profile에 있는 정보를 file에 저장
# profile_file.close()

profile_file=open('profile.pickle','rb')
profile=pickle.load(profile_file)#file에 있는 정보를 profile에 불러오기
print(profile)
profile_file.close()

