import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Получение списка всех файлов изображений в папке
image_paths = glob.glob('sphere_sfm/*.jpg')
images = []

# Загрузка всех изображений
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is not None:
        images.append(img)
    else:
        print(f"Ошибка при загрузке изображения: {image_path}")

print(f"Загружено изображений: {len(images)}")

# Инициализация детектора ключевых точек и дескрипторов
sift = cv2.SIFT_create()

# Инициализация списка для хранения ключевых точек и дескрипторов
keypoints_list = []
descriptors_list = []

# Извлечение ключевых точек и дескрипторов для каждого изображения
for img in images:
    keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

for i, (keypoints, descriptors) in enumerate(zip(keypoints_list, descriptors_list)):
    print(f"Изображение {i}: ключевых точек - {len(keypoints)}, дескрипторов - {descriptors.shape if descriptors is not None else 'None'}")

# Инициализация матчера
bf = cv2.BFMatcher()

# Инициализация списка для хранения совпадений
matches_list = []

# Поиск совпадений между последовательными изображениями
for i in range(len(images) - 1):
    matches = bf.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)
    # Применение ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    matches_list.append(good_matches)

for i, matches in enumerate(matches_list):
    print(f"Изображение {i} и {i+1}: совпадений - {len(matches)}")

# Инициализация списков для хранения точек и матриц гомографии
points_list = []
H_list = []

# Вычисление матриц гомографии между последовательными изображениями
for i in range(len(matches_list)):
    src_pts = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches_list[i]]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in matches_list[i]]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    points_list.append((src_pts, dst_pts))
    H_list.append(H)

for i, H in enumerate(H_list):
    print(f"Гомография между изображениями {i} и {i+1}: \n{H}")

# Построение траектории камеры
trajectory = [np.eye(4)]
for H in H_list:
    Rt = np.eye(4)
    Rt[:3, :3] = H[:3, :3]
    Rt[:3, 3] = H[:3, 2]
    trajectory.append(trajectory[-1] @ Rt)

for i, pose in enumerate(trajectory):
    print(f"Позиция камеры {i}: \n{pose}")

# Визуализация траектории камеры
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for pose in trajectory:
    ax.scatter(pose[0, 3], pose[1, 3], pose[2, 3])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Camera Trajectory')
plt.show()
