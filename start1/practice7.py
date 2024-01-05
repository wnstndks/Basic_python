# 메소드 오버라이딩

class unit:
    def __init__(self,name,hp,speed):
        self.name=name
        self.hp=hp
        self.speed=speed
    def move(self,location):
        print('[지상 유닛 이동]')
        print('{}:{}방향으로 이동합니다.[속도:{}]'.format(self.name,location,self.speed))
    
class attackunit(unit):
    def __init__(self,name,hp,speed,damage):
        unit.__init__(self,name,hp,speed)
        self.damage=damage
    def attack(self, location):
        print('{}:{}방향으로 적군을 공격합니다'.format(self.name,location))
    def attack(self,location): 
         print('{}:{} 방향으로 적군을 공격합니다 [공격력:{}]'.format(self.name,location,self.damage)) 

    def damaged(self,damage):
         print('{}:{}데미지를 입었습니다'.format(self.name,self.damage))
         self.hp-=damage
         print('{}:현재 체력은 {}입니다'.format(self.name,self.hp))
         if self.hp<=0:
             print('{}: 파괴되었습니다'.format(self.name))
class flyable:
     def __init__(self,flying_speed):
         self.flying_speed=flying_speed
     def fly(self,name,location):
         print('{}:{}방향으로 날아갑니다.[속도:{}]'\
            .format(name,location,self.flying_speed))
class flyableattackunit(attackunit,flyable):
     def __init__(self,name,hp,damage,flying_speed):
         attackunit.__init__(self,name,hp,0,damage) #지상 스피드는 0
         flyable.__init__(self,flying_speed)

     def move(self,location):
         print('[공중 유닛이동]')
         self.fly(self.name,location)

# 벌쳐:지상 유닛,기동성이 좋음
vulture=attackunit('벌쳐',80,10,20)

# 배틀 크루저: 공중유닛,체력 좋음,공격력도 좋음
battlecruiser=flyableattackunit('배틀크루저',500,25,3)



vulture.move('11시')
# battlecruiser.fly(battlecruiser.name,'9시')
battlecruiser.move('9시')

# 매번 우리가 지상유닛인지 공중유닛인지 확인 해야하는 것은 귀찮기 때문에 move만 써도 지상,공중 분리
# flyableattackunit에서 move함수를 새로 정의하였기때문에 날아다니는 효과를 낼수 있음


# pass
# 건물

class buildingunit(unit):
    def __init__(self, name, hp, location):
        pass #pass는 일단은 넘어간다는 의미 일단은 함수가 완성된 것처럼 만듬

# 서플라이 디폿:건물,1개 건물=8만큼의 유닛 생성
supply_depot=buildingunit('서플라이 디폿',500,'7시')
        
def game_start():
    print('[알림] 새로운 게임을 시작합니다.')
def game_over():
    pass
game_start()
game_over() #pass를 사용함으로서 그냥 넘어감




























