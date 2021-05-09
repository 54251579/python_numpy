# -*- coding: utf-8 -*-
import numpy as np # numpy를 np로 축약하여 사용

# 버전 확인(설치 확인)
print(np.__version__)


# 일반 리스트
arr1 = [i for i in range(5)]

print(arr1)      # 내용
print(type(arr1))# arr의 타입

# 리스트를 NumPy 배열로
np_arr1 = np.array(arr1)

print(np_arr1)       # 내용
print(type(np_arr1)) # np_arr의 타입

# 출력 결과
# [0, 1, 2, 3, 4]   리스트는 중괄호에 ,로 구별
# <class 'list'>
# [0 1 2 3 4]       NumPy 배열은 중괄호에 공백으로 구별
# <class 'numpy.ndarray'>

arr2 = [1, 2, 3, 4, 5.1]
np_arr2 = np.array(arr2)
print(arr2)
print(np_arr2)

# 출력 결과
# [1, 2, 3, 4, 5.1]
# [1.  2.  3.  4.  5.1]

for i in arr2:
    print(type(i))
for i in np_arr2:
    print(type(i))

# 출력 결과
# <class 'int'>
# <class 'int'>
# <class 'int'>
# <class 'int'>
# <class 'float'>
# <class 'numpy.float64'>
# <class 'numpy.float64'>
# <class 'numpy.float64'>
# <class 'numpy.float64'>
# <class 'numpy.float64'>

# 타입이 일치하지않아 정수형(하위 타입)이 실수형(상위 타입)으로 바꿔 모든 요소를 같은 타입으로 일치 시킴


arr2_1 = ['1', 2, 3, 4, 5.1]
np_arr2_1 = np.array(arr2_1)

for i in arr2_1:
    print(type(i))
for i in np_arr2_1:
    print(type(i))
# 출력 결과
# <class 'str'>
# <class 'int'>
# <class 'int'>
# <class 'int'>
# <class 'float'>
# <class 'numpy.str_'>
# <class 'numpy.str_'>
# <class 'numpy.str_'>
# <class 'numpy.str_'>
# <class 'numpy.str_'>
# 제일 상위 타입인 문자열로 변환

arr3 = [1, 2, 3, 4, 5]
np_arr3 = np.array(arr3, dtype='float64') # dtype을 통해 정수형을 실수형으로

print(arr3)
print(np_arr3)

# [1, 2, 3, 4, 5]
# [1. 2. 3. 4. 5.]

arr4 = ['3.14', 2, 5, 3]
np_arr4 = np.array(arr4, dtype='float32')

print(arr4)
print(np_arr4)

# ['3.14', 2, 5, 3]
# [3.14 2.   5.   3.  ]

# 실수로 변환 시키는 경우 3.1400001로 변환되는 경우가 있는데 3.14랑 비교할경우 ==로 비교하면 문제가 생기므로 x-3.14 >= or =< 를 하여 오차를 줄여줌

# NumPy 배열의 속성 확인
np_arr5 = np.array([i for i in range(5)])

print(np_arr5)
print(np_arr5.ndim)       # 차원의 수
print(np_arr5.shape)      # tuple로 반환 되어
print(len(np_arr5.shape)) # len으로 차원의 수 산출
print(np_arr5.size)       # 항목의 수

# [0 1 2 3 4]
# 1
# (5,)
# 1
# 5

# NumPy 배열의값 접근
nums1 = np.array([i for i in range(5)])

print(nums1[1])
print(nums1[:3])
print(nums1[2:4])
print(nums1[::2])

# 1
# [0 1 2]
# [2 3]
# [0 2 4]

# 2차원 배열
nums2 = np.array([[1,4,2],[7,5,3]])
print(nums2)
print(nums2.ndim)       # 차원의 수
print(nums2.shape)      # tuple로 반환 되어
print(len(nums2.shape)) # len으로 차원의 수 산출
print(nums2.size)       # 항목의 수


# 배열 값 접근
print(nums2[0, 2])
print(nums2[0][2])

# [x:y, w:h] x행부터 y-1행 w열 h-1열까지

print(nums2[0:1,])  # 0부터 1-1행 열 전체
print(nums2[0:1,:]) # 0부터 1-1행 열 전체
print(nums2[:,1:2]) # 행 전체 1열 2-1열
print(nums2[1,1:])  # 1행 전체 1열 부터 끝
print(nums2[0:,1:]) # 행 전체 1열 부터 끝
print(nums2[0,:])   # 0행 열 전체

nums3 = np.array(3)
print(nums3)
print(nums3.ndim)
print(nums3.shape)

# 3차원 배열
nums4 = np.array([[[1,4,2],[7,5,3]],
                    [[0,4,8],[6,9,1]],
                    [[7,6,9],[4,0,8]]])
print(nums4)

# NumPy배열의 주의사항
nums5 = np.array([i for i in range(5)])
reference = nums5[1:4]
copy_ver = nums5[1:4].copy()

print(reference)
print(copy_ver)
nums5[2]=10
print(reference)
print(copy_ver)
# copy()하지 않고 그냥 대입할 경우 참조 할당이 되기 때문에 독립적으로 사용 할 경우에는 꼭 copy()를 사용해야 한다.

# NumPy 내부함수를 통한 배열 생성
print(np.zeros((3,3)))  # 0으로 채운 NumPy배열 생성
print(np.ones((3,3)))   # 1로 채운 NumPy배열 생성
print(np.full((3,3), 4))# 사용자가 지정한 수로 채운 NumPy배열 생성

print(np.identity(4))   # size*size 크기의 단위 행렬 생성
print(np.eye(3))        # size*size 크기의 단위 행렬을 k만큼 이동시킨 행렬 생성
print(np.eye(3, k=1))
print(np.eye(3, k=-2))

print(np.random.random((2,2))) # 2,2 크기의 0~1사이의 난수를 가진 행렬 생성
print(np.random.normal((2,2))) # 2,2 크기의 정규분포를 따른 난수를 가진 행렬 생성
print(np.random.randint(0, 5, (2,2))) # 2,2 크기의 0~4의 정수형 난수를 가진 행렬 생성

# linespace(start, stop, num, endpoint, retstep)
# 시작, 끝, 원소 개수, 끝 값을 포함할건지, 스텝 값의 리턴여부
print(np.linspace(0,1,num=5,endpoint=True))
print(np.linspace(0,1,num=5,endpoint=False))
print(np.linspace(0,1,num=5,endpoint=False, retstep=True))

# np.arange(start, stop, step, dtype=?)
# start부터 stop전까지 step단위로 배열 생성
print(np.arange(0.1, 1.1, 0.1))

# np.reshape
arr_reshape_before_ex = np.arange(1, 11)
arr_reshape_after_ex = arr_reshape_before_ex.reshape((2,5))
print(arr_reshape_before_ex)
print(arr_reshape_after_ex)

# transpose(전치)
print(arr_reshape_after_ex.T)

# swapaxes(axis1, axis2)    axis1과 axis2룰 바꿈
print(arr_reshape_after_ex.swapaxes(0,1))       # .T와 같음
print(np.swapaxes(arr_reshape_after_ex, 0, 1))  # NumPy내부에 있는 함수에 배열을 넣어 변경 가능

# 3차원 배열에서 swapaxes(), transpose()

# axis의 순서는 열(0), 행(1), 층(2)의 반대 층(0), 행(1), 열(2) 순서이다
# transpose()의 매개변수로 들어가는 순서도 층(0), 행(1), 열(2)이다

arr_3_dim_swapaxes_ex = np.arange(1, 28)
arr_3_dim_swapaxes_ex = arr_3_dim_swapaxes_ex.reshape((3,3,3))
print(arr_3_dim_swapaxes_ex)

print(arr_3_dim_swapaxes_ex.swapaxes(1,2))      # 열과 행이 바뀜
print(arr_3_dim_swapaxes_ex.transpose(0, 2, 1)) # 열과 행이 바뀜

print(arr_3_dim_swapaxes_ex.swapaxes(0,1))      # 행과 층 바뀜
print(arr_3_dim_swapaxes_ex.transpose(1, 0, 2)) # 행과 층 바뀜

print(arr_3_dim_swapaxes_ex.T)
print(arr_3_dim_swapaxes_ex.transpose())        # 층과 열 바뀜
print(arr_3_dim_swapaxes_ex.swapaxes(0,2))      # 층과 열 바뀜
print(arr_3_dim_swapaxes_ex.transpose(2, 1, 0)) # 층과 열 바뀜

print(np.transpose(arr_3_dim_swapaxes_ex, (2, 1, 0))) # NumPy내부에 있는 함수에 배열을 넣어 변경 가능하나 transpose의 경우 튜플 또는 리스트여야 한다

# NumPy의 배열 연결
# concatenate()를 이용한 연결
arr_concatenate_ex1 = np.arange(10)
arr_concatenate_ex2 = np.arange(10, 20)
arr_concatenate_ex3 = np.arange(20, 30)

print(np.concatenate([arr_concatenate_ex1, arr_concatenate_ex2]))
print(np.concatenate([arr_concatenate_ex1, arr_concatenate_ex2, arr_concatenate_ex3]))

# 2차원 배열의 연결
arr_concatenate_ex4 = np.arange(10).reshape((2,5))
arr_concatenate_ex5 = np.arange(10, 20).reshape((2,5))

# 행에 연결
print(np.concatenate([arr_concatenate_ex4, arr_concatenate_ex5]))
# 열에 연결
print(np.concatenate([arr_concatenate_ex4, arr_concatenate_ex5], axis = 1))

# NumPy의 배열 연결
# vstack을 이용한 연결
arr_vstack_ex1 = np.arange(1, 4)
arr_vstack_ex2 = np.arange(4, 10).reshape((2,3))

print(np.vstack([arr_vstack_ex1, arr_vstack_ex2]))

# hstack을 이용한 연결
arr_hstack_ex1 = np.array([[1,2,3,4],[6,7,8,9]])
arr_hstack_ex2 = np.array([[5],[10]])

print(np.hstack([arr_hstack_ex1, arr_hstack_ex2]))

# NumPy배열의 분할
arr_split_ex1 = np.arange(0, 10)
arr_split_ex_result1, arr_split_ex_result2, arr_split_ex_result3 = np.split(arr_split_ex1, [3,5])
print(arr_split_ex_result1, arr_split_ex_result2, arr_split_ex_result3)

# 2차원 배열 분할
arr_split_ex2 = np.arange(16).reshape((4,4))
# 행을 분할
arr_split_ex_result4, arr_split_ex_result5 = np.split(arr_split_ex2, [2])
print(arr_split_ex_result4)
print(arr_split_ex_result5)
# 열을 분할
arr_split_ex_result6, arr_split_ex_result7 = np.split(arr_split_ex2, [2], axis=1)
print(arr_split_ex_result6)
print(arr_split_ex_result7)

# vsplit을 이용한 분할
arr_vsplit_ex_result1, arr_vsplit_ex_result2 = np.vsplit(arr_split_ex2, [2])
print(arr_vsplit_ex_result1)
print(arr_vsplit_ex_result2)

# hsplit을 이용한 분할
arr_hsplit_ex_result1, arr_hsplit_ex_result2 = np.hsplit(arr_split_ex2, [2])
print(arr_hsplit_ex_result1)
print(arr_hsplit_ex_result2)

# 범용 함수(유니버셜 함수)
ufunc_ex = np.array([4.98, 0, -2.19, 3.75, -1.98, -4.64])

# np.equal(arr1, arr2)
# np.not_equal(arr1, arr2)


print('ufunc_ex+5\t', ufunc_ex+5)
print('ufunc_ex-5\t', ufunc_ex-5)
print('ufunc_ex*5\t', ufunc_ex*5)
print('ufunc_ex/5\t', ufunc_ex/5)
print('np.around ufunc_ex\t', np.around(ufunc_ex))      # 반올림
print('np.round_ ufunc_ex\t', np.round_(ufunc_ex, 1))   # 소수점 n자리까지 반올림
print('np.rint ufunc_ex\t', np.rint(ufunc_ex))          # 가장 가까운 정수로
print('np.fix ufunc_ex\t\t', np.fix(ufunc_ex))            # 0에 가까운 방향으로
print('np.ceil ufunc_ex\t', np.ceil(ufunc_ex))          # 천장 값으로
print('np.floor ufunc_ex\t', np.floor(ufunc_ex))        # 바닥 값으로
print('np.trunc ufunc_ex\t', np.trunc(ufunc_ex))        # 소수점 제거

dimension_1 = np.arange(1, 5)
dimension_2 = np.arange(1, 5).reshape((2,2))

# NumPy는 inf(무한 최대 최소 값을 넘긴상태) NaN(Not a Number)가 있다
# np.nanprod는 NaN을 1로 바꿔 연산 np.nansum는 NaN을 0으로 바꿔 연산을 한다

print('np.prod dimension1\t\t', np.prod(dimension_1))               # 모든 원소 곱
print('np.prod dimension2\t\t', np.prod(dimension_2))               # 모든 원소 곱
print('np.prod dimension2 axis=0\t', np.prod(dimension_2, axis=0))  # 행의 원소 곱
print('np.prod dimension2 axis=1\t', np.prod(dimension_2, axis=1))  # 열의 원소 곱
print('dimension2.prod axis=1\t', dimension_2.prod(axis=1))

print('np.sum dimension1\t\t', np.sum(dimension_1))                                 # 모든 원소 합
print('np.sum dimension2\t\t', np.sum(dimension_2))                                 # 모든 원소 합
print('np.sum dimension1 keepdims=True\t\t', np.sum(dimension_1, keepdims=True))    # 모든 원소 합 차원 유지
print('np.sum dimension2 keepdims=True\t\t', np.sum(dimension_2, keepdims=True))    # 모든 원소 합 차원 유지
print('np.sum dimension2 axis=0\t', np.sum(dimension_2, axis=0))                    # 행의 원소 합
print('np.sum dimension2 axis=1\t', np.sum(dimension_2, axis=1))                    # 열의 원소 합
print('dimension2.sum axis=1\t', dimension_2.sum(axis=1))

print('np.cumprod dimension1\t\t', np.cumprod(dimension_1))               # 누적 곱
print('np.cumprod dimension2\t\t', np.cumprod(dimension_2))               # 누적 곱
print('np.cumprod dimension2 axis=0\t', np.cumprod(dimension_2, axis=0))  # 행의 누적 곱
print('np.cumprod dimension2 axis=1\t', np.cumprod(dimension_2, axis=1))  # 열의 누적 곱

print('np.cumsum dimension1\t\t', np.cumsum(dimension_1))                                 # 누적 합
print('np.cumsum dimension2\t\t', np.cumsum(dimension_2))                                 # 누적 합
print('np.cumsum dimension2 axis=0\t', np.cumsum(dimension_2, axis=0))                    # 행의 누적 합
print('np.cumsum dimension2 axis=1\t', np.cumsum(dimension_2, axis=1))                    # 열의 누적 합

diff_ex = np.array([1,2,3,5,10,13,20])
print('np.diff diff_ex\t', np.diff(diff_ex))        # 차분(2-1,3-2,5-3,10-5,13-10,20-13)

diff_ex2 = np.array([[1, 2, 4, 8], [11, 12, 14, 18]])
print('np.diff diff_ex2\t', np.diff(diff_ex2))          # 행의 차분 
print('np.diff diff_ex2\t', np.diff(diff_ex2, axis=1))  # 열의 차분
print('np.diff diff_ex2\t', np.diff(diff_ex2, axis=0))  # 행의 차분
print('np.ediff1d diff_ex2\t', np.ediff1d(diff_ex2))    # 1차원 배열로 만들어 차분

print('np.gradient diff_ex\t', np.gradient(diff_ex))          # 기울기 (2-1), ((3-2)+(2-1))/2 ... ((13-10)+(10-5))/2 ((20-13)+(13-10))/2 (20-13)
print('np.gradient diff_ex, 2\t', np.gradient(diff_ex, 2))    # 기울기 (2-1)/2, ((2-1)+(3-2))/(2*2) ... x축의 단위를 2로 늘려줌
print('np.gradient diff_ex, 2\t', np.gradient(diff_ex, edge_order=2))   # 양옆에만 2차 차분

print('np.gradient diff_ex2\t\t', np.gradient(diff_ex2))                                 # 기울기 행, 열 반환
print('np.gradient diff_ex2 axis=0\t', np.gradient(diff_ex2, axis=0))                    # 행의 기울기
print('np.gradient diff_ex2 axis=1\t', np.gradient(diff_ex2, axis=1))                    # 열의 기울기

print('np.exp dimension_1\t', np.exp(dimension_1))          # e^x인 지수함수로 변환
#print('np.log 0\t\t', np.log(0))                            # -inf 출력
print('np.log dimension_1\t', np.log(np.exp(dimension_1)))  # 밑이 e인 로그함수
print('np.log10 dimension_1\t', 10**np.log10(dimension_1))  # 밑이 10인 로그함수
print('np.log2 dimension_1\t', 2**np.log2(dimension_1))     # 밑이 2인 로그함수
print('np.log1p -1 dimension_1\t', np.exp(np.log1p(dimension_1))-1) # 밑이 e인 로그함수 (x+1)

# 0 15 30 45 60 75 90
rad = np.array([0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, 5*np.pi/12, np.pi/2])
sin_val = np.array([0, (np.sqrt(6)-np.sqrt(2))/4, 1/2, np.sqrt(2)/2, np.sqrt(3)/2, (np.sqrt(6)+np.sqrt(2))/4, 1])
cos_val = np.flip(sin_val)
tan_val = np.array([0, 2-np.sqrt(3), np.sqrt(3)/3, 1, np.sqrt(3), 2+np.sqrt(3), np.inf])
print('np.sin\t', np.sin(rad))
print('sin\t', sin_val)
print('np.cos\t', np.cos(rad))
print('cos\t', cos_val)
print('np.tan\t', np.tan(rad))
print('tan\t', tan_val)
# 삼각함수의 역함수
# np.arcsin()
# np.arccos()
# np.arctan()

print('np.rad2deg\t', np.rad2deg(rad))
print('np.rad2deg\t', np.deg2rad(np.rad2deg(rad)))
abs_ex = np.array([-1,-2,-3, -2-6j, 2-4j, -1+7j])
fabs_ex = np.array([-1,-2,-3])
sqrt_ex = np.array([2**2,5**2,7**2])

print('np.abs\t', np.abs(abs_ex))   # 복소수 됨
# print('np.fabs', np.fabs(abs_ex))
print('np.fabs\t', np.fabs(fabs_ex))  # 복소수 안됨 abs보다 빠름
print('np.sqrt\t', np.sqrt(sqrt_ex))
print('np.square\t', np.square(np.sqrt(sqrt_ex)))
print('np.mof\t', np.modf(ufunc_ex))
print('np.mof[0]\t', np.modf(ufunc_ex)[0])
print('np.mof[1]\t', np.modf(ufunc_ex)[1])
print('np.sign\t', np.sign(ufunc_ex))   # 1 양수 0 0 -1 음수

# isnan NaN을 True 나머지는 False인 배열반환
# isfinite 유한수를  True 나머지는 False인 배열반환
# isinf 무한을 True 나머지는 False인 배열반환
# isposinf 양인 무한을 True 나머지는 False인 배열반환
# isneginf 인 무한을 True 나머지는 False인 배열반환
# all 다참이면 True 아니면 False axis를 통해 행, 열을 나눠 찾을 수 있음
# any 하나라도 참이면 True 아니면 False axis를 통해 행, 열을 나눠 찾을 수 있음

# np.logical_not(조건)으로 조건에 맞으면 True 틀리면 False인 함수를 반환

# 이외에도 더있음

# 브로드 캐스팅

# 규칙
# 1.두 배열의 차원수가 다르면 작으쪽의 차원을 가진 배열 형상의 앞쪽을 1로 채운다
# 2. 두배열의 형상이 어떤 차원에서도 일치하지 않는다면  어떤 차원에서도 일치 하지 않으면 차원의 형상이 1인 배열이 다른 형상과 일치하도록 늘어난다
# 3. 임의의 차원에서 크기가 일치하지 않고 1도  아니라면 오류가 발생한다

a = np.arange(4)
b = np.arange(1, 5)
print(a)
print(b)
print(a+b)

print(np.eye(4)+0.01*np.ones((4)))

print(np.arange(3)+5)
print(np.ones((3,3))+np.arange(3))
print(np.arange(3).reshape((3,1))+np.arange(3))

# 안되는 경우
arr_b1 = np.ones((3,2))
arr_b2 = np.arange(3)
print(np.shape(arr_b1))
print(np.shape(arr_b2))
# arr_b2의 차원을 (1,3)
# (3,2)와 (3,3)이므로 불가

# 구조화된 NumPy배열
name = ['A', 'B', 'C', 'D']
age = [12, 15, 45, 56]
weight = [55.0, 84.0, 95.6, 73.2]
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'), 'formats':('U10', 'i4', 'f8')})

data['name']=name
data['age']=age
data['weight']=weight
print(data)
print(data['name'])
print(data[1])
print(data[0:2]['age'])
print(data[data['age']<30]['name'])