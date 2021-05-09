# -*- coding: utf-8 -*-
import numpy as np # numpy�� np�� ����Ͽ� ���

# ���� Ȯ��(��ġ Ȯ��)
print(np.__version__)


# �Ϲ� ����Ʈ
arr1 = [i for i in range(5)]

print(arr1)      # ����
print(type(arr1))# arr�� Ÿ��

# ����Ʈ�� NumPy �迭��
np_arr1 = np.array(arr1)

print(np_arr1)       # ����
print(type(np_arr1)) # np_arr�� Ÿ��

# ��� ���
# [0, 1, 2, 3, 4]   ����Ʈ�� �߰�ȣ�� ,�� ����
# <class 'list'>
# [0 1 2 3 4]       NumPy �迭�� �߰�ȣ�� �������� ����
# <class 'numpy.ndarray'>

arr2 = [1, 2, 3, 4, 5.1]
np_arr2 = np.array(arr2)
print(arr2)
print(np_arr2)

# ��� ���
# [1, 2, 3, 4, 5.1]
# [1.  2.  3.  4.  5.1]

for i in arr2:
    print(type(i))
for i in np_arr2:
    print(type(i))

# ��� ���
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

# Ÿ���� ��ġ�����ʾ� ������(���� Ÿ��)�� �Ǽ���(���� Ÿ��)���� �ٲ� ��� ��Ҹ� ���� Ÿ������ ��ġ ��Ŵ


arr2_1 = ['1', 2, 3, 4, 5.1]
np_arr2_1 = np.array(arr2_1)

for i in arr2_1:
    print(type(i))
for i in np_arr2_1:
    print(type(i))
# ��� ���
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
# ���� ���� Ÿ���� ���ڿ��� ��ȯ

arr3 = [1, 2, 3, 4, 5]
np_arr3 = np.array(arr3, dtype='float64') # dtype�� ���� �������� �Ǽ�������

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

# �Ǽ��� ��ȯ ��Ű�� ��� 3.1400001�� ��ȯ�Ǵ� ��찡 �ִµ� 3.14�� ���Ұ�� ==�� ���ϸ� ������ ����Ƿ� x-3.14 >= or =< �� �Ͽ� ������ �ٿ���

# NumPy �迭�� �Ӽ� Ȯ��
np_arr5 = np.array([i for i in range(5)])

print(np_arr5)
print(np_arr5.ndim)       # ������ ��
print(np_arr5.shape)      # tuple�� ��ȯ �Ǿ�
print(len(np_arr5.shape)) # len���� ������ �� ����
print(np_arr5.size)       # �׸��� ��

# [0 1 2 3 4]
# 1
# (5,)
# 1
# 5

# NumPy �迭�ǰ� ����
nums1 = np.array([i for i in range(5)])

print(nums1[1])
print(nums1[:3])
print(nums1[2:4])
print(nums1[::2])

# 1
# [0 1 2]
# [2 3]
# [0 2 4]

# 2���� �迭
nums2 = np.array([[1,4,2],[7,5,3]])
print(nums2)
print(nums2.ndim)       # ������ ��
print(nums2.shape)      # tuple�� ��ȯ �Ǿ�
print(len(nums2.shape)) # len���� ������ �� ����
print(nums2.size)       # �׸��� ��


# �迭 �� ����
print(nums2[0, 2])
print(nums2[0][2])

# [x:y, w:h] x����� y-1�� w�� h-1������

print(nums2[0:1,])  # 0���� 1-1�� �� ��ü
print(nums2[0:1,:]) # 0���� 1-1�� �� ��ü
print(nums2[:,1:2]) # �� ��ü 1�� 2-1��
print(nums2[1,1:])  # 1�� ��ü 1�� ���� ��
print(nums2[0:,1:]) # �� ��ü 1�� ���� ��
print(nums2[0,:])   # 0�� �� ��ü

nums3 = np.array(3)
print(nums3)
print(nums3.ndim)
print(nums3.shape)

# 3���� �迭
nums4 = np.array([[[1,4,2],[7,5,3]],
                    [[0,4,8],[6,9,1]],
                    [[7,6,9],[4,0,8]]])
print(nums4)

# NumPy�迭�� ���ǻ���
nums5 = np.array([i for i in range(5)])
reference = nums5[1:4]
copy_ver = nums5[1:4].copy()

print(reference)
print(copy_ver)
nums5[2]=10
print(reference)
print(copy_ver)
# copy()���� �ʰ� �׳� ������ ��� ���� �Ҵ��� �Ǳ� ������ ���������� ��� �� ��쿡�� �� copy()�� ����ؾ� �Ѵ�.

# NumPy �����Լ��� ���� �迭 ����
print(np.zeros((3,3)))  # 0���� ä�� NumPy�迭 ����
print(np.ones((3,3)))   # 1�� ä�� NumPy�迭 ����
print(np.full((3,3), 4))# ����ڰ� ������ ���� ä�� NumPy�迭 ����

print(np.identity(4))   # size*size ũ���� ���� ��� ����
print(np.eye(3))        # size*size ũ���� ���� ����� k��ŭ �̵���Ų ��� ����
print(np.eye(3, k=1))
print(np.eye(3, k=-2))

print(np.random.random((2,2))) # 2,2 ũ���� 0~1������ ������ ���� ��� ����
print(np.random.normal((2,2))) # 2,2 ũ���� ���Ժ����� ���� ������ ���� ��� ����
print(np.random.randint(0, 5, (2,2))) # 2,2 ũ���� 0~4�� ������ ������ ���� ��� ����

# linespace(start, stop, num, endpoint, retstep)
# ����, ��, ���� ����, �� ���� �����Ұ���, ���� ���� ���Ͽ���
print(np.linspace(0,1,num=5,endpoint=True))
print(np.linspace(0,1,num=5,endpoint=False))
print(np.linspace(0,1,num=5,endpoint=False, retstep=True))

# np.arange(start, stop, step, dtype=?)
# start���� stop������ step������ �迭 ����
print(np.arange(0.1, 1.1, 0.1))

# np.reshape
arr_reshape_before_ex = np.arange(1, 11)
arr_reshape_after_ex = arr_reshape_before_ex.reshape((2,5))
print(arr_reshape_before_ex)
print(arr_reshape_after_ex)

# transpose(��ġ)
print(arr_reshape_after_ex.T)

# swapaxes(axis1, axis2)    axis1�� axis2�� �ٲ�
print(arr_reshape_after_ex.swapaxes(0,1))       # .T�� ����
print(np.swapaxes(arr_reshape_after_ex, 0, 1))  # NumPy���ο� �ִ� �Լ��� �迭�� �־� ���� ����

# 3���� �迭���� swapaxes(), transpose()

# axis�� ������ ��(0), ��(1), ��(2)�� �ݴ� ��(0), ��(1), ��(2) �����̴�
# transpose()�� �Ű������� ���� ������ ��(0), ��(1), ��(2)�̴�

arr_3_dim_swapaxes_ex = np.arange(1, 28)
arr_3_dim_swapaxes_ex = arr_3_dim_swapaxes_ex.reshape((3,3,3))
print(arr_3_dim_swapaxes_ex)

print(arr_3_dim_swapaxes_ex.swapaxes(1,2))      # ���� ���� �ٲ�
print(arr_3_dim_swapaxes_ex.transpose(0, 2, 1)) # ���� ���� �ٲ�

print(arr_3_dim_swapaxes_ex.swapaxes(0,1))      # ��� �� �ٲ�
print(arr_3_dim_swapaxes_ex.transpose(1, 0, 2)) # ��� �� �ٲ�

print(arr_3_dim_swapaxes_ex.T)
print(arr_3_dim_swapaxes_ex.transpose())        # ���� �� �ٲ�
print(arr_3_dim_swapaxes_ex.swapaxes(0,2))      # ���� �� �ٲ�
print(arr_3_dim_swapaxes_ex.transpose(2, 1, 0)) # ���� �� �ٲ�

print(np.transpose(arr_3_dim_swapaxes_ex, (2, 1, 0))) # NumPy���ο� �ִ� �Լ��� �迭�� �־� ���� �����ϳ� transpose�� ��� Ʃ�� �Ǵ� ����Ʈ���� �Ѵ�

# NumPy�� �迭 ����
# concatenate()�� �̿��� ����
arr_concatenate_ex1 = np.arange(10)
arr_concatenate_ex2 = np.arange(10, 20)
arr_concatenate_ex3 = np.arange(20, 30)

print(np.concatenate([arr_concatenate_ex1, arr_concatenate_ex2]))
print(np.concatenate([arr_concatenate_ex1, arr_concatenate_ex2, arr_concatenate_ex3]))

# 2���� �迭�� ����
arr_concatenate_ex4 = np.arange(10).reshape((2,5))
arr_concatenate_ex5 = np.arange(10, 20).reshape((2,5))

# �࿡ ����
print(np.concatenate([arr_concatenate_ex4, arr_concatenate_ex5]))
# ���� ����
print(np.concatenate([arr_concatenate_ex4, arr_concatenate_ex5], axis = 1))

# NumPy�� �迭 ����
# vstack�� �̿��� ����
arr_vstack_ex1 = np.arange(1, 4)
arr_vstack_ex2 = np.arange(4, 10).reshape((2,3))

print(np.vstack([arr_vstack_ex1, arr_vstack_ex2]))

# hstack�� �̿��� ����
arr_hstack_ex1 = np.array([[1,2,3,4],[6,7,8,9]])
arr_hstack_ex2 = np.array([[5],[10]])

print(np.hstack([arr_hstack_ex1, arr_hstack_ex2]))

# NumPy�迭�� ����
arr_split_ex1 = np.arange(0, 10)
arr_split_ex_result1, arr_split_ex_result2, arr_split_ex_result3 = np.split(arr_split_ex1, [3,5])
print(arr_split_ex_result1, arr_split_ex_result2, arr_split_ex_result3)

# 2���� �迭 ����
arr_split_ex2 = np.arange(16).reshape((4,4))
# ���� ����
arr_split_ex_result4, arr_split_ex_result5 = np.split(arr_split_ex2, [2])
print(arr_split_ex_result4)
print(arr_split_ex_result5)
# ���� ����
arr_split_ex_result6, arr_split_ex_result7 = np.split(arr_split_ex2, [2], axis=1)
print(arr_split_ex_result6)
print(arr_split_ex_result7)

# vsplit�� �̿��� ����
arr_vsplit_ex_result1, arr_vsplit_ex_result2 = np.vsplit(arr_split_ex2, [2])
print(arr_vsplit_ex_result1)
print(arr_vsplit_ex_result2)

# hsplit�� �̿��� ����
arr_hsplit_ex_result1, arr_hsplit_ex_result2 = np.hsplit(arr_split_ex2, [2])
print(arr_hsplit_ex_result1)
print(arr_hsplit_ex_result2)

# ���� �Լ�(���Ϲ��� �Լ�)
ufunc_ex = np.array([4.98, 0, -2.19, 3.75, -1.98, -4.64])

# np.equal(arr1, arr2)
# np.not_equal(arr1, arr2)


print('ufunc_ex+5\t', ufunc_ex+5)
print('ufunc_ex-5\t', ufunc_ex-5)
print('ufunc_ex*5\t', ufunc_ex*5)
print('ufunc_ex/5\t', ufunc_ex/5)
print('np.around ufunc_ex\t', np.around(ufunc_ex))      # �ݿø�
print('np.round_ ufunc_ex\t', np.round_(ufunc_ex, 1))   # �Ҽ��� n�ڸ����� �ݿø�
print('np.rint ufunc_ex\t', np.rint(ufunc_ex))          # ���� ����� ������
print('np.fix ufunc_ex\t\t', np.fix(ufunc_ex))            # 0�� ����� ��������
print('np.ceil ufunc_ex\t', np.ceil(ufunc_ex))          # õ�� ������
print('np.floor ufunc_ex\t', np.floor(ufunc_ex))        # �ٴ� ������
print('np.trunc ufunc_ex\t', np.trunc(ufunc_ex))        # �Ҽ��� ����

dimension_1 = np.arange(1, 5)
dimension_2 = np.arange(1, 5).reshape((2,2))

# NumPy�� inf(���� �ִ� �ּ� ���� �ѱ����) NaN(Not a Number)�� �ִ�
# np.nanprod�� NaN�� 1�� �ٲ� ���� np.nansum�� NaN�� 0���� �ٲ� ������ �Ѵ�

print('np.prod dimension1\t\t', np.prod(dimension_1))               # ��� ���� ��
print('np.prod dimension2\t\t', np.prod(dimension_2))               # ��� ���� ��
print('np.prod dimension2 axis=0\t', np.prod(dimension_2, axis=0))  # ���� ���� ��
print('np.prod dimension2 axis=1\t', np.prod(dimension_2, axis=1))  # ���� ���� ��
print('dimension2.prod axis=1\t', dimension_2.prod(axis=1))

print('np.sum dimension1\t\t', np.sum(dimension_1))                                 # ��� ���� ��
print('np.sum dimension2\t\t', np.sum(dimension_2))                                 # ��� ���� ��
print('np.sum dimension1 keepdims=True\t\t', np.sum(dimension_1, keepdims=True))    # ��� ���� �� ���� ����
print('np.sum dimension2 keepdims=True\t\t', np.sum(dimension_2, keepdims=True))    # ��� ���� �� ���� ����
print('np.sum dimension2 axis=0\t', np.sum(dimension_2, axis=0))                    # ���� ���� ��
print('np.sum dimension2 axis=1\t', np.sum(dimension_2, axis=1))                    # ���� ���� ��
print('dimension2.sum axis=1\t', dimension_2.sum(axis=1))

print('np.cumprod dimension1\t\t', np.cumprod(dimension_1))               # ���� ��
print('np.cumprod dimension2\t\t', np.cumprod(dimension_2))               # ���� ��
print('np.cumprod dimension2 axis=0\t', np.cumprod(dimension_2, axis=0))  # ���� ���� ��
print('np.cumprod dimension2 axis=1\t', np.cumprod(dimension_2, axis=1))  # ���� ���� ��

print('np.cumsum dimension1\t\t', np.cumsum(dimension_1))                                 # ���� ��
print('np.cumsum dimension2\t\t', np.cumsum(dimension_2))                                 # ���� ��
print('np.cumsum dimension2 axis=0\t', np.cumsum(dimension_2, axis=0))                    # ���� ���� ��
print('np.cumsum dimension2 axis=1\t', np.cumsum(dimension_2, axis=1))                    # ���� ���� ��

diff_ex = np.array([1,2,3,5,10,13,20])
print('np.diff diff_ex\t', np.diff(diff_ex))        # ����(2-1,3-2,5-3,10-5,13-10,20-13)

diff_ex2 = np.array([[1, 2, 4, 8], [11, 12, 14, 18]])
print('np.diff diff_ex2\t', np.diff(diff_ex2))          # ���� ���� 
print('np.diff diff_ex2\t', np.diff(diff_ex2, axis=1))  # ���� ����
print('np.diff diff_ex2\t', np.diff(diff_ex2, axis=0))  # ���� ����
print('np.ediff1d diff_ex2\t', np.ediff1d(diff_ex2))    # 1���� �迭�� ����� ����

print('np.gradient diff_ex\t', np.gradient(diff_ex))          # ���� (2-1), ((3-2)+(2-1))/2 ... ((13-10)+(10-5))/2 ((20-13)+(13-10))/2 (20-13)
print('np.gradient diff_ex, 2\t', np.gradient(diff_ex, 2))    # ���� (2-1)/2, ((2-1)+(3-2))/(2*2) ... x���� ������ 2�� �÷���
print('np.gradient diff_ex, 2\t', np.gradient(diff_ex, edge_order=2))   # �翷���� 2�� ����

print('np.gradient diff_ex2\t\t', np.gradient(diff_ex2))                                 # ���� ��, �� ��ȯ
print('np.gradient diff_ex2 axis=0\t', np.gradient(diff_ex2, axis=0))                    # ���� ����
print('np.gradient diff_ex2 axis=1\t', np.gradient(diff_ex2, axis=1))                    # ���� ����

print('np.exp dimension_1\t', np.exp(dimension_1))          # e^x�� �����Լ��� ��ȯ
#print('np.log 0\t\t', np.log(0))                            # -inf ���
print('np.log dimension_1\t', np.log(np.exp(dimension_1)))  # ���� e�� �α��Լ�
print('np.log10 dimension_1\t', 10**np.log10(dimension_1))  # ���� 10�� �α��Լ�
print('np.log2 dimension_1\t', 2**np.log2(dimension_1))     # ���� 2�� �α��Լ�
print('np.log1p -1 dimension_1\t', np.exp(np.log1p(dimension_1))-1) # ���� e�� �α��Լ� (x+1)

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
# �ﰢ�Լ��� ���Լ�
# np.arcsin()
# np.arccos()
# np.arctan()

print('np.rad2deg\t', np.rad2deg(rad))
print('np.rad2deg\t', np.deg2rad(np.rad2deg(rad)))
abs_ex = np.array([-1,-2,-3, -2-6j, 2-4j, -1+7j])
fabs_ex = np.array([-1,-2,-3])
sqrt_ex = np.array([2**2,5**2,7**2])

print('np.abs\t', np.abs(abs_ex))   # ���Ҽ� ��
# print('np.fabs', np.fabs(abs_ex))
print('np.fabs\t', np.fabs(fabs_ex))  # ���Ҽ� �ȵ� abs���� ����
print('np.sqrt\t', np.sqrt(sqrt_ex))
print('np.square\t', np.square(np.sqrt(sqrt_ex)))
print('np.mof\t', np.modf(ufunc_ex))
print('np.mof[0]\t', np.modf(ufunc_ex)[0])
print('np.mof[1]\t', np.modf(ufunc_ex)[1])
print('np.sign\t', np.sign(ufunc_ex))   # 1 ��� 0 0 -1 ����

# isnan NaN�� True �������� False�� �迭��ȯ
# isfinite ���Ѽ���  True �������� False�� �迭��ȯ
# isinf ������ True �������� False�� �迭��ȯ
# isposinf ���� ������ True �������� False�� �迭��ȯ
# isneginf �� ������ True �������� False�� �迭��ȯ
# all �����̸� True �ƴϸ� False axis�� ���� ��, ���� ���� ã�� �� ����
# any �ϳ��� ���̸� True �ƴϸ� False axis�� ���� ��, ���� ���� ã�� �� ����

# np.logical_not(����)���� ���ǿ� ������ True Ʋ���� False�� �Լ��� ��ȯ

# �̿ܿ��� ������

# ��ε� ĳ����

# ��Ģ
# 1.�� �迭�� �������� �ٸ��� �������� ������ ���� �迭 ������ ������ 1�� ä���
# 2. �ι迭�� ������ � ���������� ��ġ���� �ʴ´ٸ�  � ���������� ��ġ ���� ������ ������ ������ 1�� �迭�� �ٸ� ����� ��ġ�ϵ��� �þ��
# 3. ������ �������� ũ�Ⱑ ��ġ���� �ʰ� 1��  �ƴ϶�� ������ �߻��Ѵ�

a = np.arange(4)
b = np.arange(1, 5)
print(a)
print(b)
print(a+b)

print(np.eye(4)+0.01*np.ones((4)))

print(np.arange(3)+5)
print(np.ones((3,3))+np.arange(3))
print(np.arange(3).reshape((3,1))+np.arange(3))

# �ȵǴ� ���
arr_b1 = np.ones((3,2))
arr_b2 = np.arange(3)
print(np.shape(arr_b1))
print(np.shape(arr_b2))
# arr_b2�� ������ (1,3)
# (3,2)�� (3,3)�̹Ƿ� �Ұ�

# ����ȭ�� NumPy�迭
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