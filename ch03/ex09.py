"""
PIL 패키지와 numpy 패키지를 이용하면,
이미지 파일(jpg, png, bmp, ...)의 픽셀 정보를 numpy.ndarray 형식으로 변환하거나
numpy.ndarray 형식의 이미지 픽셀 정보를 이미지 파일로 저장할 수 있습니다.
"""
import numpy as np
from PIL import Image


def image_to_pixel(image_file):
    """이미지 파일 이름(경로)를 파라미터로 전달받아서,
    numpy.ndarray에 픽셀 정보를 저장해서 리턴."""
    img = Image.open(image_file, mode='r')    # 이미지 파일 오픈
    print('type: ',type(img))
    pixels = np.array(img)    # 이미지 파일 객체를 numpy.ndarray 형식으로 변환
    print('pixels shape: ', pixels.shape)

    return pixels


def pixel_to_image(pixel, image_file):
    """numpy.ndarray 형식의 이미지 픽셀 정보와, 저장할 파일 이름을 파라미터로
    전달받아서, 이미지 파일을 저장"""
    img = Image.fromarray(pixel)   # ndarray 타입의 데이터를 이미지로 변환
    print('type: ',type(img))
    img.show()      # 이미지 뷰어를 사용해서 이미지 보기
    img.save(image_file)   # 이미지 객체를 파일로 저장


if __name__ == '__main__':
    # image_to_pixel(), pixel_to_image() 함수 테스트
    pixels1 = image_to_pixel('크리스마스.jpg')    # shape: (497, 860, 3)  -> (세로 길이, 가로 길이, color)
    pixels2 = image_to_pixel('크리스마스트리.png')  # shape: (1208, 860, 3)

    pixel_to_image(pixels1, 'test.jpg')
    pixel_to_image(pixels2, 'test2.png')