# with->파일을 열고, 닫는 것을 편하게 할 수 있음, 
# 열었던 파일에 대해서 따로 close를 할 필요가 없음
# import pickle

# with open('profile.pickle','rb') as profile_file:
#     print(pickle.load(profile_file))

# with open('study.txt','w',encoding='utf8') as study_file:
#     study_file.write('파이썬을 열심히 공부하고 있어요')

# with open('study.txt','r',encoding='utf8') as study_file:
#     print(study_file.read())

# for i in range(1,51):
#     with open(str(i)+'주차.txt','w',encoding='utf8') as report_file:
#         report_file.write('-{}주차 주간보고-'.format(i))
#         report_file.write('\n부서:')
#         report_file.write('\n이름:')
#         report_file.write('\n업무요약:')

# w모드 일때는 파일이 있더라도 덮어쓰기 가능



# 클래스=서로 연관되어있는 변수와 함수의 집합
# 마린: 공격 유닛,총을 쏠수 있음,군인

# name='마린' #유닛의 이름
# hp=40 #유닛의 체력
# damage=5 #유닛의 공격력

# print('{}유닛이 생성되었습니다'.format(name))
# print('체력{}, 공격력{}\n'.format(hp,damage))

# # 탱크:공격 유닛,탱크,포를 쏠수 있는데,일반 모드/시즈모드

# tank_name='탱크'
# tank_hp=150
# tank_damage=35

# print('{}유닛이 생성되었습니다'.format(tank_name))
# print('체력{}, 공격력{}\n'.format(tank_hp,tank_damage))

# def attack(name,location,damage):
#     print('{}:{}방향으로 적군을 공격합니다[공격력{}]'.format\
#         (name, location,damage))

# attack(name,'1시',damage)
# attack(tank_name,'1시',tank_damage)


class unit:
     def __init__(self,name,hp,damage):#필요한 값들을 정의해주는 것
         self.name=name
         self.hp=hp
         self.damage=damage
         print('{}유닛이 생성되었습니다.'.format(self.name))
         print('체력 {},공격력 {}.'.format(self.hp,self.damage))


# marine1=unit('마린',40,5) #self를 제외한 나머지
# marine2=unit('마린',40,5)
# tank=unit('탱크',150,35)

# # 하나의 클래스를 통해서 서로 다른 마린과 탱크 유닛이 생성 가능해짐


# __init__  =파이썬에서 쓰이는 생성자->객체(마린이나 탱크같이 
# 어떤 클래스로부터 만들어지는 녀석들)가 만들어질 때 자동으로 호출되는 부분
# 객체가 생성될 때는 init함수에 정의된 개수와 동일한 개수만큼 값을 부과해야 된다.


# 멤버 변수->클래스 내에서 정의된 변수 그 변수를 가지고 우리가 초기화 및 사용가능 
# ex self.name,self.hp 등

# 레이스:공중유닛,비행기,클로킹(상대방에게 보이지 않음)


wraith1=unit('레이스',80,5)
print('유닛 이름:{},공격력:{}'.format(wraith1.name,wraith1.damage)) #.을 찍을시
# 뒤에 적을수 있는 멤버 변수를 알수 있고 접근 가능


# 마인드 컨트롤:상대방 유닛을 내것으로 만드는 것(빼앗음)
wraith2=unit('빼앗은 레이스',80,5)
wraith2.clocking=True #clocking 이라는 기능이개발됨-->class에는 없지만 외부에서 멤버변수를 추가로 할당할 수 있다. 
# 그러나 wraith1에는 clocking이라는 변수가 없음, 확장된 변수는 내가 확장을 한 객체에 대해서만 적용되고 
# 기존의 객체에 대해서는 적용되지 않음

if wraith2.clocking==True:
    print('{}는 현재 클로킹 상태입니다.'.format(wraith2.name))






















