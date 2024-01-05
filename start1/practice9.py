from random import *

class unit:
    def __init__(self,name,hp,speed):
        self.name=name
        self.hp=hp
        self.speed=speed
        print('{}유닛이 생성되었습니다.'.format(name))

    def move(self,location):
        print('{} : {} 방향으로 이동합니다. [속도: {}]'.format(self.name,location,self.speed))

    def damaged(self,damage):
        print('{} : {} 데미지를 입었습니다.'.format(self.name,self.damage))
        self.hp-=damage
        print('{} : 현재 체력은 {}입니다.'.format(self.name,self.hp))
        if self.hp<=0:
            print('{} : 파괴되었습니다..'.format(self.name))

# 공격 유닛

class attackunit(unit):
    def __init__(self,name,hp,speed,damage):
        unit.__init__(self,name,hp,speed)
        self.damage=damage
    def attack(self,location):
        print('{} : {} 방향으로 적군을 공격합니다. [공격력: {}]'.format(self.name,location,self.damage))

#마린 
class marine(attackunit):
    def __init__(self):
        attackunit.__init__(self,'마린',40,1,5)    

    # 스팀팩: 일정 시간동안 이동 및 공격 속도 증가,자기 체력 10 감소
    def stimpack(self):
        if self.hp>=10:
            self.hp-=10
            print('{}: 스팀팩을 사용합니다. (hp 10감소)'.format(self.name))
        else:
            print('{}: 체력이 부족하여 스팀팩을 사용하지 않습니다.'.format(self.name))

class tank(attackunit):
    # 시즈모드: 탱크 지상에 고정시켜, 더 높은 공격력,이동 불가
    seize_developed=False  # 시즈모드 개발여부

    def __init__(self):
        attackunit.__init__(self,'탱크',150,1,35)
        self.seize_mode=False    

    def set_seize_mode(self):
        if tank.seize_developed==False:
            return 
            
        # 현재 시즈모드가 아닐때 ->시즈 모드
        if self.seize_mode==False:
            print('{} : 시즈모드로 전환합니다.'.format(self.name)) 
            self.damage*=2
            self.seize_mode=True

        # 현재 시즈모드 일때->시즈모드 해제
        else:
            print('{} : 시즈모드를 해제합니다.'.format(self.name)) 
            self.damage/=2
            self.seize_mode=False
        
class flyable:
    def __init__(self,flying_speed):
        self.flying_speed=flying_speed
    def fly(self,name,location):
        print('{} : {} 방향으로 날아갑니다. [속도:{}]'.format(self.name,location,self.flying_speed))

# 공중 공격 유닛 클래스
class flyableattackunit(attackunit,flyable):
    def __init__(self,name,hp,damage,flying_speed):
        attackunit.__init__(self,name,hp,0,damage)
        flyable.__init__(self,flying_speed)

    def move(self,location):
        self.fly(self.name,location)

# 레이스
class wraith(flyableattackunit):
    def __init__(self):
        flyableattackunit.__init__(self,'레이스',80,20,5)
        self.clocked=False #클로킹 모드(해제 상태)
    
    def clocking(self):
        if self.clocked==True:
            print('{}: 클로킹 모드를 해제합니다.'.format(self.name))
            self.clocked=False
        else:
            print('{}: 클로킹 모드를 설정합니다.'.format(self.name))
            self.clocked=True



def game_start():
    print('[알림] 새로운 게임을 시작합니다.')

def game_over():
    print('player: gg') #good game
    print('[player]님이 게임에서 퇴장하셨습니다.')


# 실제 게임 시작
game_start()

# 유닛 생성
m1=marine()
m2=marine()
m3=marine()

t1=tank()
t2=tank()

w1=wraith()


# 유닛 일괄 관리
attack_units=[]
attack_units.append(m1)
attack_units.append(m2)
attack_units.append(m3)
attack_units.append(t1)
attack_units.append(t2)
attack_units.append(w1)


# 전군 이동
for unit in attack_units:
    unit.move('1시')

# 탱크 시즈 모드
tank.seize_developed=True
print('[알림] 탱크 시즈모드 개발이 완료되었습니다')

# 공격모드 준비(마린: 스팀팩, 탱크:시즈 모드, 레이스:클로킹)
for unit in attack_units:
    if isinstance(unit,marine): #어떤 객체가 특정 클래스의 인스탠스인지 확인하는 것 현재 유닛이 마린인지 확인하는 것
        unit.stimpack()

    elif isinstance(unit,tank):
        unit.set_seize_mode()
    
    elif isinstance(unit,wraith):
        unit.clocking()

# 전군 공격
for unit in attack_units:
    unit.attack('1시')

# 전군 피해
for unit in attack_units:
    unit.damaged(randint(5,21)) #공격은 랜덤으로 받음(5~20)

# 게임 종료
game_over()

from random import *

class House:
    def __init__(self,location,house_type,deal_type,price,completion_year):
        self.location=location
        self.house_type=house_type
        self.deal_type=deal_type
        self.price=price
        self.completion_year=completion_year

    def show_detail(self):
        print('{} {} {} {} {}'.format(self.location,self.house_type,self.deal_type,self.price,self.completion_year))


x=3
print('총 {}대의 매물이 있습니다.'.format(x))

list=[]

for i in range(x):
    locat=input('위치: ')
    ht=input('타입: ')
    dt=input('매매 타입: ')    
    p=input('가격: ')    
    cy=input('완공 년도: ')
    house1=House(locat,ht,dt,p,cy)
    list.append(house1)    

for home in list:
    home.show_detail()
