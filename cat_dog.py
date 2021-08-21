import os, glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import cv2
import PIL.Image as pilimg
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential


# # 분류 대상 카테고리 선택하기
accident_dir = "./dogs-vs-cats/train"
categories = ["cat","dog"]
nb_classes = len(categories)
##이미지 크기 지정
image_w = 64
image_h = 64
pixels = image_w * image_h * 3
# 이미지 데이터 읽어 들이기
X = []
Y = []
for idx, cat in enumerate(categories):
    # 레이블 지정
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지
    image_dir = accident_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = pilimg.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)      # numpy 배열로 변환
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)
# 학습 전용 데이터와 테스트 전용 데이터 구분
X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)

print('저장중...')
np.save("./dogs-vs-cats/ffff.npy", xy)
print("저장끝, ", len(Y))

#######################################################################

categories = ["cat", "dog"]
nb_classes = len(categories)

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

image_w = 64
image_h = 64
# 데이터 불러오기
X_train, X_test, y_train, y_test = np.load("./dogs-vs-cats/ffff.npy")
# 데이터 정규화하기(0~1사이로)
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float") / 256
print('X_train shape:', X_train.shape)

# 모델 구조 정의
model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# 전결합층
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(256))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# 모델 구축하기
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer='rmsprop',
    metrics=['accuracy'])
# 모델 확인
#print(model.summary())

# 학습 완료된 모델 저장
hdf5_file = "./dogs-vs-cats/model.hdf5"
if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)

else:
    # 학습한 모델이 없으면 파일로 저장
    model.fit(X_train, y_train, batch_size=32, epochs=20)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    print('\n테스트 정확도:', test_acc, '\n테스트 로스:', test_loss)

    model.save_weights(hdf5_file)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.legend(['training', 'validation'], loc = 'upper left')
    # plt.show()



##############################################################################
#
def enn(img1):
    img1 = cv2.resize(img1, dsize=(64, 64), interpolation=cv2.INTER_AREA)
    data = np.asarray(img1)
    X = np.array(data)
    X = X.astype("float") / 256
    X = X.reshape(-1, 64, 64, 3)

    return X

cap = cv2.VideoCapture(0) #카메라로 불러오기

while True:

    _, img = cap.read()

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    X = enn(img)

    pred = model.predict(X)
    result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
    print('예측 값 : ', categories[result[0]])


    cv2.imshow('img', img)
    if cv2.waitKey(1  ) == ord('q'):
        sys.exit(1)
