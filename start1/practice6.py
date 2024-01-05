# # # 일반 유닛
# # class unit:
# #     def __init__(self,name,hp,damage):
# #         self.name=name
# #         self.hp=hp
# #         self.damage=damage
# #         print('{}유닛이 생성되었습니다.'.format(self.name))
# #         print('체력{},공격력{}'.format(self.hp,self.damage))

# # # 공격 유닛
# # class attackunit:
# #     def __init__(self,name,hp,damage):
# #         self.name=name #왼쪽의 그냥 일반 name은 전달받은 인자를 쓴다는 것 ,self.name은 class내 변수에 접근 가능함 즉,자기 자신에 있는 멤버 변수에 접근 가능
# #         self.hp=hp
# #         self.damage=damage
# #         print('{}유닛이 생성되었습니다.'.format(self.name))
# #         print('체력{},공격력{}'.format(self.hp,self.damage))

# #     def attack(self,location): #self는 자기 자신을 의미, class내에서 method 앞에는 self를 무조건 써야함, self를 통해서 자기 자신의 변수에 접근 가능,
# #         print('{}:{} 방향으로 적군을 공격합니다 [공격력:{}]'.format(self.name,location,self.damage)) #그냥 location은 전달받은 인자를 쓴것

# #     def damaged(self,damage):
# #         print('{}:{}데미지를 입었습니다'.format(self.name,self.damage))
# #         self.hp-=damage
# #         print('{}:현재 체력은 {}입니다'.format(self.name,self.hp))
# #         if self.hp<=0:
# #             print('{}: 파괴되었습니다'.format(self.name))

# # 파이어뱃:공격 유닛,화염방사기 
# # firebat1=attackunit('파이어뱃',50,16)
# # firebat1.attack('5시')

# # firebat1.damaged(25)
# # firebat1.damaged(25)

# # 상속=물려받는 것
# # 메딕:의무병->공격력 없음

# # 일반 유닛
# class unit:
#     def __init__(self,name,hp):
#         self.name=name
#         self.hp=hp
# # 공격 유닛->유닛이라는 클래스를 상속받아 어택 유닛을 만듬 일반유닛의 멤버변수 그대로 어택 유닛에서 쓸수 있음
# class attackunit(unit):
#     def __init__(self,name,hp,damage):
#         unit.__init__(self,name,hp) #유닛에서 만들어진 생성자를 호출해서 이 클래스에서도 쓰일수 있음 
#         self.damage=damage

#     def attack(self,location): #self는 자기 자신을 의미, class내에서 method 앞에는 self를 무조건 써야함, self를 통해서 자기 자신의 변수에 접근 가능,
#         print('{}:{} 방향으로 적군을 공격합니다 [공격력:{}]'.format(self.name,location,self.damage)) #그냥 location은 전달받은 인자를 쓴것

#     def damaged(self,damage):
#         print('{}:{}데미지를 입었습니다'.format(self.name,self.damage))
#         self.hp-=damage
#         print('{}:현재 체력은 {}입니다'.format(self.name,self.hp))
#         if self.hp<=0:
#             print('{}: 파괴되었습니다'.format(self.name))

# firebat1=attackunit('파이어뱃',50,16)
# firebat1.attack('5시')

# firebat1.damaged(25)
# firebat1.damaged(25)


# # 다중 상속->부모 클래스를 두개 이상 상속 받는 것 unit=부모 attackunit=자식==>다중상속:부모가 둘 이상

# # 드랍쉽:공중 유닛,수송기,마린 파이어뱃 탱크 등을 수송하는 유닛 공격 기능 없음

# # 날 수 있는 기능을 가진 클래스
# class flyable:
#     def __init__(self,flying_speed):
#         self.flying_speed=flying_speed
#     def fly(self,name,location):
#         print('{}:{}방향으로 날아갑니다.[속도:{}]'\
#             .format(name,location,self.flying_speed))

# # 공중 공격 유닛 클래스->어택 유닛 플라이어블 클래스를 상속 받음 두곳에서 제공하는 모든 것을 사용가능
# class flyableattackunit(attackunit,flyable):
#     def __init__(self,name,hp,damage,flying_speed):
#         attackunit.__init__(self,name,hp,damage) #self는 항상 넣어야 함
#         flyable.__init__(self,flying_speed)


# # 발키리: 공중 공격 유닛,한번에 14발 미사일 발사
# valkyrie=flyableattackunit('발키리',200,6,5)
# valkyrie.fly(valkyrie.name,'3시')#flyable에 있는 fly 함수 호출,flyable 같은 경우 name 변수가 없으므로 별도로 name 추가, 속도만 정보로 가지고 있음


