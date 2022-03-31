import cv2
import json
import numpy as np
import os
import sys

def make_white_black(image):
	by_horizontal = (image[1:, :] - image[:-1, :])[:, :-1]
	by_vertical = (image[:, 1:] - image[:, :-1])[:-1:, :]

	wb_res = np.sqrt(np.linalg.norm(by_horizontal, axis=-1) ** 2 + np.linalg.norm(by_vertical, axis=-1) ** 2)
	return wb_res

def calculate_integral_sum(image):
	wight, height = image.shape[0], image.shape[1]
	sum_image = np.zeros_like(image)

	for i in range(wight):
		for j in range(height):
			current_sum = image[i, j]

			if i > 0:
				current_sum += sum_image[i - 1, j]
			if j > 0:
				current_sum += sum_image[i, j - 1]
			if i > 0 and j > 0:
				current_sum -= sum_image[i - 1, j - 1]

			sum_image[i, j] = current_sum

	return sum_image

def calculate_nearest(height, near_top, near_smthg, is_left = True):
	stack = []

	range_ = range(height)

	if not is_left:
		range_ = reversed(range_)

	for j in range_:
		while stack and near_top[stack[-1]] <= near_top[j]:
			stack.pop()
		if stack:
			near_smthg[j] = stack[-1]
		stack.append(j)

def get_sum(rect, sum_image):
	top_sum, left_sum, top_left_sum = 0, 0, 0
	if rect[0][1] > 0:
		top_sum = sum_image[rect[1][0], rect[0][1]]

	if rect[0][0] > 0:
		left_sum = sum_image[rect[0][0], rect[1][1]]

	if rect[0][0] > 0 and rect[0][1] > 0:
		top_left_sum = sum_image[rect[0][0], rect[0][1]]

	return np.sum((sum_image[rect[1][0], rect[1][1]] - top_sum) + (top_left_sum - left_sum))


def detect_color_gradient(wb_image, integral_sum_image, config):
	wight, height = wb_image.shape[0], wb_image.shape[1]
	near_top, near_left = -(np.ones(height).astype(int)), -(np.ones(height).astype(int))
	near_right = (np.ones(height).astype(int) * height)

	answer_rect = None
	max_rect_area = 0
	max_grad, min_avg = config["grad_max"], config["min_avg"]

	for i in range(wight):
		for j in range(height):
			if wb_image[i][j] > max_grad:
				near_top[j] = i
	
		calculate_nearest(height, near_top, near_left)
		calculate_nearest(height, near_top, near_right, is_left = False)

		for j in range(height):
			rect = ((near_top[j] + 1, near_left[j] + 1), (i, near_right[j] - 1))
			area = (rect[1][0] - rect[0][0]) * (rect[1][1] - rect[0][1])
			if area > 0:
				rect_avg = get_sum(rect, integral_sum_image) / area
				if area > max_rect_area and rect_avg > min_avg:
					max_rect_area, answer_rect = area, rect

	return answer_rect


def find_gradient_rect(image_dir, image_name, image, config):
	print('The search for the gradient in the picture {0} has begun.'.format(image_name))

	wb_image = make_white_black(image)
	integral_sum_image = calculate_integral_sum(wb_image)

	answer = detect_color_gradient(wb_image, integral_sum_image, config)
	return answer

def draw_result_image(image, found_rect, dir_new_image, name_old_image, config):
	image = cv2.rectangle(
		image, (found_rect[0][1], found_rect[0][0]), (found_rect[1][1], found_rect[1][0]),
		color=config["rect_colour"], thickness=config["rect_thickness"]
	)

	if not os.path.exists(dir_new_image):
		os.mkdir(dir_new_image)

	dst_path = os.path.join(dir_new_image, os.path.split(name_old_image)[-1])
	cv2.imwrite(dst_path, image)

	print('The gradient in the picture {0} is found.'.format(name_old_image))

def load_config(config_file_name):
	config = {}
	with open(config_file_name, 'r') as config_file:
		config = json.load(config_file)
	return config

def main():

	if len(sys.argv) != 2:
		print('Missing argument - configuration file name.')
		sys.exit()

	if not os.path.exists(sys.argv[1]):
		print('Wrong path to config file {0}.'.format(sys.argv[1]))
		sys.exit()

	config = load_config(sys.argv[1])

	all_images = config["images"]
	images_dir, dir_for_results = config['images_dir'], config["dir_for_answer"]

	for image_name in all_images:
		image_path = images_dir + '/' + image_name
		if not os.path.exists(image_path):
			print('Wrong path to image {0}.'.format(image_name))
			continue

		image = cv2.imread(image_path)

		if image is None:
			print('Image {0} doesn\'t load.'.format(image_name))
			continue

		image = image.astype(np.float64)
		rect = find_gradient_rect(images_dir, image_name, image, config)

		if not rect is None:
			draw_result_image(image, rect, dir_for_results, image_name, config)
		else:
			print('Gradient areas could not be found in the picture {0}. Try changing the setting in the config file.'.format(image_name))

		print()

if __name__ == '__main__':
	main()