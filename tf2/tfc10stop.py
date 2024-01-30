# 모델 처리 도중 사요외는 클래스를 별도 모듈로 작성 후 호출
# 연습 대상으로 callback 클래스를 상속받은 임의의 클래스 작성
import keras

class MyEarlyStop(keras.callbacks.Callback):
    def on_epoch_end(self, epoch,logs={}):
        if logs.get('loss') < 0.25:
            print('\n 학습을 조기 종료 합니다.')
            #loss값이 0.25보다 작아질때 뭔가를 많이 할수 있다.
            #....
            self.model.stop_training = True


            