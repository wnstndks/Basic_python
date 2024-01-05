# class unit:
#     def __init__(self,name,hp,speed):
#         self.name=name
#         self.hp=hp
#         self.speed=speed
#     def move(self,location):
#         print('[지상 유닛 이동]')
#         print('{}:{}방향으로 이동합니다.[속도:{}]'.format(self.name,location,self.speed))
    
# class attackunit(unit):
#     def __init__(self,name,hp,speed,damage):
#         unit.__init__(self,name,hp,speed)
#         self.damage=damage
#     def attack(self, location):
#         print('{}:{}방향으로 적군을 공격합니다'.format(self.name,location))
#     def attack(self,location): 
#          print('{}:{} 방향으로 적군을 공격합니다 [공격력:{}]'.format(self.name,location,self.damage)) 

#     def damaged(self,damage):
#          print('{}:{}데미지를 입었습니다'.format(self.name,self.damage))
#          self.hp-=damage
#          print('{}:현재 체력은 {}입니다'.format(self.name,self.hp))
#          if self.hp<=0:
#              print('{}: 파괴되었습니다'.format(self.name))
# class flyable:
#      def __init__(self,flying_speed):
#          self.flying_speed=flying_speed
#      def fly(self,name,location):
#          print('{}:{}방향으로 날아갑니다.[속도:{}]'\
#             .format(name,location,self.flying_speed))
# class flyableattackunit(attackunit,flyable):
#      def __init__(self,name,hp,damage,flying_speed):
#          attackunit.__init__(self,name,hp,0,damage) #지상 스피드는 0
#          flyable.__init__(self,flying_speed)

#      def move(self,location):
#          print('[공중 유닛이동]')
#          self.fly(self.name,location)

# class buildingunit(unit):
#     def __init__(self, name, hp, location):
#         # unit.__init__(self,name,hp,0)
#         super().__init__(name,hp,0) #위에 문장과 동일한 것 unit을 통해서 상속받을 수 있고 
#         # super을 통해서 상속받을 수 있는데 super에서는 init 괄호안에 self 넣지 않음
#         self.location=location


class unit:
    def __init__(self):
        print('unit 생성자')

class flyable:
    def __init__(self):
        print('flyable 생성자')

class flyableunit(flyable,unit):
    def __init__(self):
        super().__init__() #super는 다중상속을 할 때는 상속받는 클래스의 순서에 따라 init함수가 호출이 되므로 따로 명시적으로 unit.__init__을 통해 두번 초기화 하는 방식을 사용하거나 unit.__init__만 사용하자



# 드랍쉽->운송 유닛

dropship=flyableunit()



































